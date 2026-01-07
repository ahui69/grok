import micropip
await micropip.install('httpx')
import httpx
import json

url = "http://77.42.73.96:8080/v1/chat/completions"  # Zakładam path jak w source
headers = {
    "Content-Type": "application/json",
    "Authorization": "Bearer dummy_key"  # Zmień na realny jeśli trzeba
}
payload = {
    "model": "neural_memory",
    "messages": [
        {"role": "system", "content": "Test pamięci"},
        {"role": "user", "content": "Insert: Test wspomnienie - działa zajebiście! Query: Co pamiętasz o teście?"}
    ],
    "temperature": 0.7
}

try:
    with httpx.Client(timeout=10) as client:
        response = client.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "No content")
        print("Odpowiedź z serwera:", content)
        print("Status:", response.status_code)
        print("Full data:", json.dumps(data, indent=2))
except Exception as e:
    print("Error podczas fetcha:", str(e))
