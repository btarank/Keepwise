// static/js/recorder.js  (patched — prefers per_speaker and hides global prediction)
(() => {
  const recordBtn = document.getElementById('recordBtn');
  const stopBtn = document.getElementById('stopBtn');
  const uploadBtn = document.getElementById('uploadBtn');
  const statusEl = document.getElementById('status');
  const visualEl = document.getElementById('visual');
  const historyEl = document.getElementById('history');
  const serverBlockInner = document.getElementById('serverBlockInner');

  let mediaRecorder = null;
  let audioChunks = [];
  let lastBlob = null;

  function escapeHtml(s) {
    if (s == null) return '';
    return String(s).replace(/[&<>"'`=\/]/g, function(ch) {
      return ({
        '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;','/':'&#x2F;','`':'&#x60;','=':'&#x3D;'
      })[ch];
    });
  }

  function setStatus(text) {
    if (statusEl) statusEl.textContent = `Status: ${text}`;
  }

  function renderHistory() {
    try {
      const hist = JSON.parse(localStorage.getItem('keepwise_uploads') || '[]');
      if (!hist.length) {
        historyEl.textContent = 'No uploads yet.';
        return;
      }
      historyEl.innerHTML = hist.slice().reverse().map(h => {
        const t = new Date(h.timestamp).toLocaleString();
        return `<div style="margin-bottom:8px"><strong>${escapeHtml(h.filename)}</strong><div style="color:#6b7280;font-size:12px">${t}</div></div>`;
      }).join('');
    } catch (e) {
      historyEl.textContent = 'No uploads yet.';
      console.error('Failed to render history', e);
    }
  }

  function setControls({ recording = false, hasBlob = false } = {}) {
    if (recordBtn) recordBtn.disabled = recording;
    if (stopBtn) stopBtn.disabled = !recording;
    if (uploadBtn) uploadBtn.disabled = !hasBlob;
  }

  function showBlobPreview(blob) {
    lastBlob = blob;
    const url = URL.createObjectURL(blob);
    if (visualEl) {
      visualEl.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:8px">
          <audio controls src="${url}"></audio>
        </div>
      `;
    }
    if (serverBlockInner) serverBlockInner.innerHTML = 'No response yet.';
    setControls({ recording: false, hasBlob: true });
  }

  // NEW: helper to build per-speaker pre block (exact format requested)
  // buildPerSpeakerPre — shows sentiment label (and score) AND per-speaker probability (if available)
function buildPerSpeakerPre(perSpeakerObj) {
  if (!perSpeakerObj || typeof perSpeakerObj !== 'object') return null;
  const keys = Object.keys(perSpeakerObj).sort((a,b)=>{
    const na = (a.match(/(\d+)$/)||[])[1]||a;
    const nb = (b.match(/(\d+)$/)||[])[1]||b;
    return Number(na)-Number(nb);
  });
  if (!keys.length) return null;

  const pre = document.createElement('pre');
  pre.style.background = '#0b1220';
  pre.style.color = '#e6edf3';
  pre.style.padding = '12px';
  pre.style.borderRadius = '8px';
  pre.style.fontFamily = 'monospace';
  pre.style.whiteSpace = 'pre-wrap';
  pre.style.fontSize = '13px';
  pre.style.lineHeight = '1.6';

  const lines = keys.map(k => {
    const sp = perSpeakerObj[k] || {};

    // sentiment object (if provided)
    const labObj = Array.isArray(sp.sentiment) && sp.sentiment[0] ? sp.sentiment[0] : (sp.sentiment && typeof sp.sentiment === 'object' ? sp.sentiment : null);
    const lab = labObj ? String(labObj.label || '').toUpperCase() : '';
    const sentScore = labObj && labObj.score !== undefined && !isNaN(Number(labObj.score)) ? Number(labObj.score) : null;
    const sentScoreStr = (sentScore !== null) ? `(${sentScore.toFixed(2)})` : '';

    // speaker-level probability (some servers use "probability" or "prob")
    let spProb = null;
    if (sp.probability !== undefined && !isNaN(Number(sp.probability))) spProb = Number(sp.probability);
    else if (sp.prob !== undefined && !isNaN(Number(sp.prob))) spProb = Number(sp.prob);

    const spProbStr = (spProb !== null) ? `prob: ${spProb.toFixed(3)}` : '';

    // Decision logic: prefer sentiment label/score to determine textual prediction, fallback to numeric prediction/prob
    let pred = 'likely to stay';
    if (labObj) {
      const l = String(labObj.label || '').toLowerCase();
      const s = Number(labObj.score || 0);
      if (l.includes('neg') && s >= 0.5) pred = 'likely to leave';
      else if (l.includes('pos') && s >= 0.5) pred = 'likely to stay';
      else pred = s >= 0.5 ? 'likely to stay' : 'likely to leave';
    } else if (sp.prediction === 0 || sp.prediction === 1) {
      pred = sp.prediction === 1 ? 'likely to leave' : 'likely to stay';
    } else if (spProb !== null) {
      pred = spProb > 0.5 ? 'likely to leave' : 'likely to stay';
    }

    const name = String(k).replace(/^SPEAKER_/i,'speaker_');
    // build HTML parts
    const speakerHtml = `<span style="color:#f472b6;font-weight:700">${escapeHtml(name)}</span>`;
    const sentimentHtml = lab ? `<span style="color:#fb7185;font-weight:700"> : sentiment: </span><span style="color:#f472b6">${escapeHtml(lab)}</span>` : `<span style="color:#fb7185;font-weight:700"> : sentiment: </span><span style="color:#6b7280">N/A</span>`;
    const scorePart = sentScore !== null ? `<span style="color:#93c5fd"> ${escapeHtml(sentScoreStr)}</span>` : '';
    const probPart = spProb !== null ? `<span style="color:#f97316"> • ${escapeHtml(spProbStr)}</span>` : '';
    const predColor = pred.includes('leave') ? '#f59e0b' : '#16a34a';
    const predHtml = `<div style="margin-top:4px"><span style="color:${predColor}; font-weight:700">prediction: ${escapeHtml(pred)}</span></div>`;

    // final assembled block for this speaker (two-line look: first line has speaker+sentiment+score+prob, second line prediction)
    return `${speakerHtml}${sentimentHtml}${scorePart}${probPart}\n${predHtml}`;
  }).join('\n\n');

  pre.innerHTML = lines;
  return pre;
}


  // Updated renderServerSummary: prefer per_speaker and hide global prediction when per_speaker exists
  function renderServerSummary(json) {
    const container = serverBlockInner;
    if (!container) return;

    // If server returned per_speaker object, render that and DO NOT show global prediction
    if (json && json.per_speaker && typeof json.per_speaker === 'object' && Object.keys(json.per_speaker).length) {
      // Clear container then insert per-speaker pre
      container.innerHTML = '';
      const perPre = buildPerSpeakerPre(json.per_speaker);
      if (perPre) {
        const wrapper = document.createElement('div');
        wrapper.style.marginBottom = '8px';
        wrapper.appendChild(perPre);
        container.appendChild(wrapper);
      }
      // Also attach a raw-json toggle so raw data remains accessible
      const rawJson = escapeHtml(JSON.stringify(json, null, 2));
      const rawId = 'raw-json-' + Date.now();
      const toggle = document.createElement('a');
      toggle.href = '#';
      toggle.textContent = 'View raw server response';
      toggle.style.display = 'inline-block';
      toggle.style.marginTop = '8px';
      toggle.style.color = '#0b63ff';
      container.appendChild(toggle);
      const pre = document.createElement('pre');
      pre.id = rawId;
      pre.style.display = 'none';
      pre.style.whiteSpace = 'pre-wrap';
      pre.style.background = '#0f1724';
      pre.style.color = '#e6edf3';
      pre.style.padding = '10px';
      pre.style.borderRadius = '6px';
      pre.style.marginTop = '8px';
      pre.style.overflow = 'auto';
      pre.style.maxHeight = '240px';
      pre.textContent = rawJson;
      container.appendChild(pre);
      toggle.addEventListener('click', (ev) => {
        ev.preventDefault();
        if (pre.style.display === 'none') { pre.style.display = 'block'; toggle.textContent = 'Hide raw response'; }
        else { pre.style.display = 'none'; toggle.textContent = 'View raw server response'; }
      });
      return;
    }

    // Fallback to older behaviour if no per_speaker
    if (!json) {
      container.innerHTML = `<div style="color:#6b7280">Server returned no JSON.</div>`;
      return;
    }

    // Defensive extraction of fields (global)
    const prediction = json.prediction !== undefined ? json.prediction : (json.pred || null);
    const probability = json.attrition_proba !== undefined ? json.attrition_proba : (json.probability !== undefined ? json.probability : (json.prob || null));
    const transcription = (json.transcription || json.transcript || json.text || null);
    const sentiment = json.sentiment || (json.readable_fields && json.readable_fields.sentiment) || null;

    let html = `<div style="padding:6px">`;

    if (prediction !== null) {
      const label = (prediction === 1 || prediction === '1' || prediction === true) ? '⚠️ Employee likely to leave' : '✅ Employee likely to stay';
      html += `<div style="font-weight:700;margin-bottom:6px">${escapeHtml(label)}</div>`;
    }

    if (probability !== null) {
      html += `<div style="color:#374151;margin-bottom:6px">Probability: <strong>${escapeHtml(String(probability))}</strong></div>`;
    }

    if (transcription) {
      const snippet = escapeHtml(String(transcription)).slice(0, 1000);
      html += `<div style="color:#374151;margin-bottom:8px"><strong>Transcript (truncated):</strong></div><div style="color:#334155;font-size:14px;line-height:1.4">${snippet}${String(transcription).length>1000 ? '…' : ''}</div>`;
    }

    if (sentiment) {
      let sentText = '';
      try {
        if (Array.isArray(sentiment)) {
          sentText = sentiment.map(s => {
            if (typeof s === 'string') return s;
            if (s.label && s.score !== undefined) return `${s.label} (${Number(s.score).toFixed(2)})`;
            return JSON.stringify(s);
          }).join(', ');
        } else if (typeof sentiment === 'object') {
          sentText = JSON.stringify(sentiment);
        } else {
          sentText = String(sentiment);
        }
      } catch (e) {
        sentText = String(sentiment);
      }
      html += `<div style="margin-top:8px;color:#374151">Sentiment: <strong>${escapeHtml(sentText)}</strong></div>`;
    }

    // Raw JSON toggle (global fallback)
    const rawJson = escapeHtml(JSON.stringify(json, null, 2));
    const rawId = 'raw-json-' + Date.now();
    html += `<div style="margin-top:10px"><a href="#" id="toggleRaw" style="font-size:13px;color:#0b63ff">View raw server response</a>
      <pre id="${rawId}" style="display:none;white-space:pre-wrap;background:#0f1724;color:#e6edf3;padding:10px;border-radius:6px;margin-top:8px;overflow:auto;max-height:240px">${rawJson}</pre></div>`;

    html += `</div>`;

    container.innerHTML = html;

    const toggle = document.getElementById('toggleRaw');
    const pre = document.getElementById(rawId);
    if (toggle && pre) {
      toggle.addEventListener('click', (ev) => {
        ev.preventDefault();
        pre.style.display = pre.style.display === 'none' ? 'block' : 'none';
        toggle.textContent = pre.style.display === 'none' ? 'View raw server response' : 'Hide raw response';
      });
    }
  }

  async function startRecording() {
    try {
      setStatus('requesting microphone permission…');
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorder = new MediaRecorder(stream);
      audioChunks = [];

      mediaRecorder.addEventListener('dataavailable', e => {
        if (e.data && e.data.size > 0) audioChunks.push(e.data);
      });

      mediaRecorder.addEventListener('stop', () => {
        const blob = new Blob(audioChunks, { type: 'audio/webm' });
        showBlobPreview(blob);
        setStatus('recording stopped. Ready to upload.');
      });

      mediaRecorder.start();
      setStatus('recording…');
      if (visualEl) visualEl.textContent = 'Recording…';
      setControls({ recording: true, hasBlob: false });
    } catch (err) {
      console.error('Recorder start error', err);
      setStatus('microphone access denied or error');
      if (visualEl) visualEl.textContent = 'Microphone access required';
    }
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === 'recording') {
      mediaRecorder.stop();
      mediaRecorder.stream && mediaRecorder.stream.getTracks().forEach(t => t.stop());
    }
  }

  async function uploadRecording() {
    if (!lastBlob) {
      setStatus('no recording to upload');
      return;
    }

    setStatus('uploading…');
    uploadBtn.disabled = true;
    recordBtn.disabled = true;
    stopBtn.disabled = true;

    let filename = `meeting_${Date.now()}.webm`;
    let resp = null;
    let respText = '';

    try {
      const form = new FormData();
      form.append('audio', lastBlob, filename);

      try {
        resp = await fetch('/meeting_upload', { method: 'POST', body: form });
      } catch (networkErr) {
        console.error('Upload failed (network):', networkErr);
        setStatus('upload failed — check console');
        if (serverBlockInner) serverBlockInner.innerHTML = `<div style="color:#b91c1c">Upload error: ${escapeHtml(networkErr.message || 'Failed to fetch')}</div>`;
        return;
      }

      respText = await resp.text().catch(err => {
        console.error('Failed reading response text:', err);
        return `Unable to read response: ${String(err)}`;
      });

      if (!resp.ok) {
        console.error('Upload failed (server):', resp.status, respText);
        setStatus('upload failed — server error');
        if (serverBlockInner) serverBlockInner.innerHTML = `<div style="color:#b91c1c">Upload failed: ${resp.status} — ${escapeHtml(respText)}</div>`;
        return;
      }

      // Insert raw server response text into the block
      if (serverBlockInner) serverBlockInner.innerHTML = respText;
      else console.log('Server response text:', respText);

      // Immediately trigger the per-speaker UI transformer (if present)
      if (window.applyPerSpeakerToServerBlock) {
        try { window.applyPerSpeakerToServerBlock(); }
        catch (e) { console.error('applyPerSpeakerToServerBlock() failed:', e); }
      }

      // Try parsing JSON for building a nicer summary; prefer JSON.per_speaker
      let json = null;
      try {
        json = JSON.parse(respText);
      } catch (e) {
        const s = respText.indexOf('{'), eidx = respText.lastIndexOf('}');
        if (s !== -1 && eidx !== -1 && eidx > s) {
          try { json = JSON.parse(respText.slice(s, eidx+1)); } catch (e2) { json = null; }
        }
      }

      setStatus('upload complete');

      const hist = JSON.parse(localStorage.getItem('keepwise_uploads') || '[]');
      hist.push({ timestamp: new Date().toISOString(), filename });
      localStorage.setItem('keepwise_uploads', JSON.stringify(hist.slice(-200)));

      renderHistory();

      // Render summary — this function now prefers per_speaker (and hides global prediction in that case)
      if (json) renderServerSummary(json);

    } catch (err) {
      console.error('UploadRecording: error', err);
      setStatus('upload failed — check console');
      if (serverBlockInner && (!serverBlockInner.textContent || serverBlockInner.textContent.trim() === '')) {
        serverBlockInner.innerHTML = `<div style="color:#b91c1c">Upload error: ${escapeHtml(err.message || String(err))}</div>`;
      }
    } finally {
      setControls({ recording: false, hasBlob: !!lastBlob });
      uploadBtn.disabled = !lastBlob;
      recordBtn.disabled = false;
      stopBtn.disabled = true;
    }
  }

  // Wire UI
  if (recordBtn) recordBtn.addEventListener('click', startRecording);
  if (stopBtn) stopBtn.addEventListener('click', stopRecording);
  if (uploadBtn) uploadBtn.addEventListener('click', uploadRecording);

  // Initial state
  setStatus('idle');
  setControls({ recording: false, hasBlob: false });
  renderHistory();

  // Keyboard shortcuts
  document.addEventListener('keydown', e => {
    if (e.key === 'r' || e.key === 'R') { if (!recordBtn.disabled) startRecording(); }
    if (e.key === 's' || e.key === 'S') { if (!stopBtn.disabled) stopRecording(); }
    if (e.key === 'u' || e.key === 'U') { if (!uploadBtn.disabled) uploadRecording(); }
  });
})();
