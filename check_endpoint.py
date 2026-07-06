import os
import json
import requests

url = os.environ["ENDPOINT_URL"]
token = os.environ["DBR_TOKEN"]

question = {"id": "Q", "type": "mcq", "text": "?",
            "correct_answer": "Paris", "max_marks": 2}
answer = {"id": "A", "question_id": "Q",
          "student_id": "s", "answer_text": "Paris"}

payload = {
    "dataframe_split": {
        "columns": ["question_json", "answer_json"],
        "data": [[json.dumps(question), json.dumps(answer)]],
    }
}

resp = requests.post(
    url,
    headers={"Authorization": f"Bearer {token}",
             "Content-Type": "application/json"},
    json=payload,
)

print("STATUS:", resp.status_code)
print("BODY:", resp.text)