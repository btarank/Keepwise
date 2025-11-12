# test_predict.py
import requests

url = "http://127.0.0.1:5000/predict"  # change to your server URL
data = {
    'age': '34',
    'business_travel': 'Travel_Rarely',
    'department': 'Research & Development',
    'monthly_income': '7000',
    'overtime': 'Yes',
    'job_role': 'Research Scientist',
    'job_satisfaction': '3',
    'total_working_years': '8',
    'years_at_company': '4',
    'environment_satisfaction': '3',
    'work_life_balance': '3',
    'performance_rating': '3'
}

r = requests.post(url, data=data)
print(r.status_code)
print(r.json())
