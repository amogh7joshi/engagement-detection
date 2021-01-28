#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from __future__ import print_function

import os
import sys
import pickle

from googleapiclient.discovery import build
from googleapiclient import errors
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.auth.exceptions import RefreshError

def authenticate():
    """Authenticate google drive, either through saved credentials or secrets."""
    SCOPES = ['https://www.googleapis.com/auth/drive']
    creds = None

    # If credentials already exist, use them.
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            try:
                creds = pickle.load(token)
            except RefreshError:
                print("Token has been expired or revoked, loading new token.")

    # Otherwise, prompt to log in and create new credentials.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port = 0)
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port = 0)
        # Save the credentials for the next run
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)

    return creds

# Authenticate Google Drive.
creds = authenticate()

# Build Service.
service = build('drive', 'v3', credentials = creds)

# Call the Google Drive API to list files.
results = service.files().list(pageSize = 1000, fields = "nextPageToken, files(id, name)").execute()
items = results.get('files', [])

# Confirm deletion of files.
confirm = input("Are you sure that you want to delete files? [yes|no]\n")
if not confirm == 'yes':
    print("You do not want to delete files, so exiting.")
    sys.exit(0)
else:
    print("You do want to delete files, confirmed.")
    print("\n--------------------------------------\n")

# Delete files that do not meet necessary criteria.
files_deleted = 0
for item in items:
    if item['name'].endswith('.hdf5'): # If file is a weights file.
        itemname = item['name'][:-5]
        model_accuracy = itemname[-4:]
        if not model_accuracy.isdigit(): # If file is from a training session.
            continue
        if not int(model_accuracy) // 100 >= 62: # If accuracy is less than 62%.
            try:
                print(f"Deleting file {item['name']}.")
                service.files().delete(fileId = item['id']).execute()
            except errors.HttpError as error:
                print(f"An error occurred while trying to delete file {item['name']}.")
                raise error
            else:
                files_deleted += 1 # A way to determine how many files were deleted.

# Inform me how many files were deleted.
print(f"{files_deleted} files deleted.")







