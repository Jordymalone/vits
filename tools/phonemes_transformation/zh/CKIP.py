import json
import requests

def call_ckip(sentence_list):
    token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzUxMiJ9.eyJ2ZXIiOjAuMSwiaWF0IjoxNjYyODA1MTExLCJ1c2VyX2lkIjoiMjkzIiwiaWQiOjQ2Niwic2NvcGVzIjoiMCIsInN1YiI6IiIsImlzcyI6IkpXVCIsInNlcnZpY2VfaWQiOiIxIiwiYXVkIjoid21ta3MuY3NpZS5lZHUudHciLCJuYmYiOjE2NjI4MDUxMTEsImV4cCI6MTY3ODM1NzExMX0.C1WpV4-9GxjT5qV7c1zZDStMmD3K6WGcIHZA7wrppOd_T2f_tpOcnRFKxdMGeCMBFYWIThh0HeXdjbkQ4AEZZ7SSddAn2YiOtguyVDHEyEYYeUerK_junGCXixexQ4jsM3Top8tMR6fsgHeE_jBH4KFSgeTlZLTkJMnw0ghAJYE"
    res = requests.post("http://140.116.245.157:2001", data={"data":sentence_list, "token":token})
    response = json.loads(res.text)

    return_lst = []
    for ws, pos in zip(response["ws"][0], response["pos"][0]):
        return_lst.append((ws, pos))
    
    # print(return_lst)
    return return_lst

if __name__ == "__main__":
    input_str = ["爸爸"]
    print(call_ckip(input_str))
