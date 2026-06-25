from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://www.googleapis.com/auth/drive']

gauth = GoogleAuth()
gauth.credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'credentials.json', scope
)
drive = GoogleDrive(gauth)

# List files in your folder
folder_id = '1zLJgkYkrM1LRgtl7v5sfAQ4aits9zfUq'
file_list = drive.ListFile({'q': f"'{folder_id}' in parents"}).GetList()

print("Connected! Files in folder:")
for f in file_list:
    print(f['title'])