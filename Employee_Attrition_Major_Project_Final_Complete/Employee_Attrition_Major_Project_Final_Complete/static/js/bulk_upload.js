// static/js/bulk_upload.js (fixed, full file)
(() => {
  // DOM elements
  const fileInput = document.getElementById('csvFile');
  const startBtn = document.getElementById('startBtn');
  const cancelBtn = document.getElementById('cancelBtn');
  const downloadLink = document.getElementById('downloadLink');
  const progressBar = document.getElementById('progressBar');
  const progressText = document.getElementById('progressText');
  const resultsArea = document.getElementById('resultsArea');

  if (!fileInput || !startBtn || !cancelBtn || !progressBar || !progressText || !resultsArea) {
    console.error('Bulk uploader: required DOM elements missing', {
      fileInput: !!fileInput, startBtn: !!startBtn, cancelBtn: !!cancelBtn,
      progressBar: !!progressBar, progressText: !!progressText, resultsArea: !!resultsArea
    });
    return;
  }

  // Required headers (must match training / backend expected column names)
  const REQUIRED_HEADERS = [
    'Age','BusinessTravel','Department','MonthlyIncome','OverTime','JobRole',
    'JobSatisfaction','TotalWorkingYears','YearsAtCompany',
    'EnvironmentSatisfaction','WorkLifeBalance','PerformanceRating'
  ];

  let controller = null;         // AbortController for cancel
  let isRunning = false;
  let stopRequested = false;

  // Minimal robust CSV parser supporting quoted fields and commas
  function parseCSV(text) {
    const rows = [];
    let cur = '';
    let row = [];
    let inQuotes = false;

    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      const nxt = text[i+1];

      if (ch === '"' ) {
        if (inQuotes && nxt === '"') { // escaped quote
          cur += '"';
          i++; // skip next
        } else {
          inQuotes = !inQuotes;
        }
        continue;
      }

      if (ch === ',' && !inQuotes) {
        row.push(cur);
        cur = '';
        continue;
      }

      // handle CRLF or LF line endings
      if ((ch === '\n' || ch === '\r') && !inQuotes) {
        row.push(cur);
        rows.push(row);
        row = [];
        cur = '';
        // handle CRLF by skipping \n when \r found
        if (ch === '\r' && nxt === '\n') i++;
        continue;
      }

      cur += ch;
    }

    // push last field
    if (cur !== '' || inQuotes) {
      row.push(cur);
    }
    if (row.length) rows.push(row);

    return rows;
  }

  // Convert array-of-arrays into array of objects given header row
  function rowsToObjects(rows) {
    if (!rows.length) return { headers: [], rows: [] };
    const headers = rows[0].map(h => h.trim());
    const objs = [];
    for (let r = 1; r < rows.length; r++) {
      const row = rows[r];
      // skip empty rows
      if (row.length === 1 && row[0].trim() === '') continue;
      const obj = {};
      for (let c = 0; c < headers.length; c++) {
        obj[headers[c]] = (row[c] !== undefined) ? row[c].trim() : '';
      }
      objs.push(obj);
    }
    return { headers, rows: objs };
  }

  // Validate presence of required headers (case-sensitive match expected by training script)
  function validateHeaders(headers) {
    const missing = REQUIRED_HEADERS.filter(h => !headers.includes(h));
    return { ok: missing.length === 0, missing };
  }

  // Utility: create CSV text from results array (array of objects)
  function toCSV(rows) {
    if (!rows.length) return '';
    const cols = Object.keys(rows[0]);
    const escape = v => {
      if (v === null || v === undefined) return '';
      const s = String(v);
      if (s.includes('"') || s.includes(',') || s.includes('\n')) {
        return `"${s.replace(/"/g,'""')}"`;
      }
      return s;
    };
    const lines = [cols.join(',')];
    for (const r of rows) {
      lines.push(cols.map(c => escape(r[c])).join(','));
    }
    return lines.join('\n');
  }

  // Show a simple message in resultsArea
  function showMessage(html) {
    resultsArea.innerHTML = `<div style="padding:8px;color:#374151">${html}</div>`;
  }

  // Initial UI state
  startBtn.disabled = !fileInput.files || fileInput.files.length === 0;
  cancelBtn.disabled = true;
  downloadLink.style.display = 'none';
  progressBar.style.width = '0%';
  progressText.textContent = 'No job running';

  // File chooser handler
  fileInput.addEventListener('change', () => {
    downloadLink.style.display = 'none';
    resultsArea.innerHTML = '';
    const f = fileInput.files[0];
    startBtn.disabled = !f;
    console.info('CSV chosen:', f ? f.name : 'none');
  });

  // Main batch start
  startBtn.addEventListener('click', async (ev) => {
    ev.preventDefault();
    const file = fileInput.files[0];
    if (!file) {
      alert('Please select a CSV file first.');
      return;
    }

    // read file
    let text;
    try {
      text = await file.text();
    } catch (e) {
      console.error('Failed to read file', e);
      showMessage('Failed to read file in browser. Try a different CSV or smaller file.');
      return;
    }

    const rawRows = parseCSV(text);
    const parsed = rowsToObjects(rawRows);
    const headers = parsed.headers;
    const rows = parsed.rows;

    // Validate header length
    const v = validateHeaders(headers);
    if (!v.ok) {
      showMessage(`Missing required headers: ${v.missing.join(', ')}. Please provide the correct CSV format.`);
      return;
    }
    if (!rows.length) {
      showMessage('CSV has no data rows.');
      return;
    }

    // Prepare UI
    isRunning = true;
    stopRequested = false;
    controller = new AbortController();
    startBtn.disabled = true;
    cancelBtn.disabled = false;
    downloadLink.style.display = 'none';
    progressBar.style.width = '0%';
    progressText.textContent = `0 / ${rows.length} processed`;
    resultsArea.innerHTML = '';
    const results = [];

    // Process sequentially to avoid overloading backend and to keep ordering
    for (let i = 0; i < rows.length; i++) {
      if (stopRequested) break;
      const row = rows[i];

      progressText.textContent = `${i} / ${rows.length} processed`;

      // Build FormData for this row: ensure keys match headers exactly
      const form = new FormData();
      for (const h of headers) {
        form.append(h, row[h] !== undefined ? row[h] : '');
      }

      // POST to /predict (same endpoint your form uses)
      try {
        const resp = await fetch('/predict', {
          method: 'POST',
          body: form,
          signal: controller.signal
        });

        if (!resp.ok) {
          // capture text error
          const textErr = await resp.text().catch(() => `HTTP ${resp.status}`);
          results.push({ ...row, prediction: '', probability: '', error: `HTTP ${resp.status}: ${textErr}` });
        } else {
          // Try to parse JSON first, otherwise handle HTML fallback
          let json = null;
          const ct = resp.headers.get('content-type') || '';

          if (ct.includes('application/json')) {
            try {
              json = await resp.json();
            } catch (e) {
              json = { raw_text: 'Invalid JSON from server' };
            }
          } else {
            // likely HTML response - parse text and try to extract useful pieces
            const txt = await resp.text();
            // Attempt JSON.parse from text (in case server embeds JSON)
            try {
              json = JSON.parse(txt);
            } catch (_) {
              // Defensive extraction from plain HTML:
              // 1) search for a Probability: 0.123 pattern
              // 2) search for common prediction label text
              let prob = '';
              let predLabel = '';
              // look for "Probability: 0.123" (case-insensitive)
              const probRegex1 = /Probability[:\s]*([0-9]*\.?[0-9]+)/i;
              const probRegex2 = /probability['"]?\s*[:=]\s*([0-9]*\.?[0-9]+)/i;
              const p1 = txt.match(probRegex1) || txt.match(probRegex2);
              if (p1) prob = p1[1];

              // look for likely/stay/leave phrases used by templates
              const predRegex = /(⚠️\s*Employee is likely to leave|Employee likely to leave|Employee is likely to leave|✅\s*Employee is likely to stay|likely to stay|likely to leave)/i;
              const pm = txt.match(predRegex);
              if (pm) {
                const found = pm[0];
                if (/leave/i.test(found)) predLabel = 1;
                else predLabel = 0;
              }

              // fallback: find first floating number that looks like a probability
              if (!prob) {
                const firstNum = txt.match(/([0-9]*\.[0-9]{1,})/);
                if (firstNum) prob = firstNum[1];
              }

              json = {
                prediction: (predLabel !== '' ? predLabel : ''),
                probability: (prob !== '' ? prob : ''),
                raw_text: txt
              };
            }
          }

          // Normalize fields defensively
          const prediction = (json && (json.pred !== undefined ? json.pred : (json.prediction !== undefined ? json.prediction : (json.prediction_text !== undefined ? json.prediction_text : ''))));
          const probability = (json && (json.attrition_proba !== undefined ? json.attrition_proba : (json.probability !== undefined ? json.probability : (json.prob !== undefined ? json.prob : ''))));

          results.push({ ...row, prediction: prediction !== undefined ? prediction : '', probability: probability !== undefined ? probability : '', raw: json });
        }
      } catch (err) {
        if (err.name === 'AbortError') {
          results.push({ ...row, prediction: '', probability: '', error: 'Canceled' });
          break;
        } else {
          results.push({ ...row, prediction: '', probability: '', error: err.message });
        }
      }

      // update progress bar
      const pct = Math.round(((i+1) / rows.length) * 100);
      progressBar.style.width = pct + '%';
      progressText.textContent = `${i+1} / ${rows.length} processed`;
    }

    // job finished / canceled
    isRunning = false;
    startBtn.disabled = false;
    cancelBtn.disabled = true;

    // ---- robust final rendering (REPLACE the old final block with this) ----
    console.log('Bulk upload: raw results array', results);

    // Build a CSV for download: flatten original headers plus result columns (defensive)
    const outputRows = results.map(r => {
      const out = {};
      // keep original input fields (headers variable still in scope)
      // headers variable is available from earlier scope
      for (const h of (typeof headers !== 'undefined' ? headers : [])) out[h] = r[h];
      // append prediction/prob fields if present
      out.prediction = r.prediction !== undefined ? r.prediction : (r.pred !== undefined ? r.pred : '');
      out.probability = r.probability !== undefined ? r.probability : (r.attrition_proba !== undefined ? r.attrion_proba : (r.probability !== undefined ? r.probability : ''));
      out.error = r.error || '';
      // include raw JSON if present (stringified)
      if (r.raw) {
        try { out.raw = JSON.stringify(r.raw); } catch(e){ out.raw = String(r.raw); }
      } else {
        out.raw = '';
      }
      return out;
    });

    console.log('Bulk upload: normalized outputRows (first 10)', outputRows.slice(0,10));

    // create CSV and download link
    const csvText = toCSV(outputRows);
    const blob = new Blob([csvText], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    downloadLink.href = url;
    const fname = (file && file.name ? file.name.replace(/\.csv$/i, '') : 'results') + '_results.csv';
    downloadLink.download = fname;
    downloadLink.style.display = 'inline-flex';
    downloadLink.textContent = 'Download results';

    // Summary count
    const successCount = outputRows.filter(r => r.error === '' && r.prediction !== '').length;
    showMessage(`Batch finished. ${successCount} / ${outputRows.length} succeeded. Download the results using the link.`);

    // Build a robust sample table (use union of keys across rows)
    function buildSampleTable(rows, maxRows=20) {
      if (!rows || rows.length === 0) return null;
      // compute union of keys
      const keys = Array.from(rows.reduce((set,row) => {
        Object.keys(row).forEach(k => set.add(k));
        return set;
      }, new Set()));
      // create table
      const table = document.createElement('table');
      table.id = 'bulk-predictions-table';

      table.style.width = '100%';
      table.style.borderCollapse = 'collapse';
      table.style.marginTop = '8px';
      const thead = table.createTHead();
      const thr = thead.insertRow();
      keys.forEach(k => {
        const th = document.createElement('th');
        th.innerText = k;
        th.style.border = '1px solid #f0f3f7';
        th.style.padding = '6px';
        th.style.background = '#fafafa';
        thr.appendChild(th);
      });
      const tbody = table.createTBody();
      const sampleRows = rows.slice(0, maxRows);
      for (const r of sampleRows) {
        const tr = tbody.insertRow();
        for (const k of keys) {
          const td = tr.insertCell();
          let v = r[k];
          if (v === null || v === undefined) v = '';
          else if (typeof v === 'object') v = JSON.stringify(v);
          td.innerText = String(v);
          td.style.border = '1px solid #f0f3f7';
          td.style.padding = '6px';
          td.style.maxWidth = '200px';
          td.style.overflow = 'hidden';
        }
      }
      return table;
    }

    resultsArea.innerHTML = '<div style="margin-top:8px"><strong>Sample results</strong></div>';
    const sampleTable = buildSampleTable(outputRows, 20);
    if (sampleTable) {
      resultsArea.appendChild(sampleTable);
    
      // trigger the DOM cleanup script (if present) to transform the "raw" column
if (window.runBulkCleanup) window.runBulkCleanup();

    } else {
      resultsArea.innerHTML += '<div style="margin-top:8px;color:#6b7280">No results to display. Check console for details.</div>';
    }

    // finalize UI
    progressBar.style.width = '100%';
    progressText.textContent = `Completed: ${outputRows.length} rows processed`;

    // ---- end replacement block ----

  });

  // Cancel button
  cancelBtn.addEventListener('click', () => {
    if (!isRunning) return;
    stopRequested = true;
    if (controller) controller.abort();
    showMessage('Cancel requested — finishing current request if any and stopping.');
    cancelBtn.disabled = true;
  });

})();
