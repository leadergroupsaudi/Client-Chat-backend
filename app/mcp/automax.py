import os
import json
import base64
import requests
from datetime import datetime
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()


class Automax3:
    def __init__(self):
        self.base_url = os.getenv("AUTOMAX_BASEURL")
        self.userid = os.getenv("AUTOMAX_USERNAME")
        self.password = os.getenv("AUTOMAX_PASSWORD")
        self.namespaceid = os.getenv("AUTOMAX_NAMESPACE")
        self.moduleid = os.getenv("AUTOMAX_MODULE")

        if not self.base_url:
            raise RuntimeError("ENV not loaded. Check .env file path.")

        self.token = None

    def login(self):
        url = f"{self.base_url}/auth/oauth2/token"

        data = {
            "grant_type": "client_credentials",
            "scope": "profile api",
        }

        headers = {
            "Content-Type": "application/x-www-form-urlencoded"
        }

        auth = HTTPBasicAuth(self.userid, self.password)

        r = requests.post(url, data=data, headers=headers, auth=auth)

        if r.status_code != 200:
            raise RuntimeError(f"Startup login failed: {r.text}")

        response = r.json()
        self.token = response["access_token"]

        print("✅ Automax login successful")

    def _headers(self):
        if not self.token:
            raise RuntimeError("Automax client used before login")

        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

    # -----------------------------
    # READ APIs
    # -----------------------------

    def get_classifications(self):
        r = requests.get(
            f"{self.base_url}/api/classifications/hierarchy",
            headers=self._headers(),
        )
        r.raise_for_status()

        data = r.json()
        print("📂 Classifications:\n", json.dumps(data, indent=2))

        return data

    def get_locations(self):
        r = requests.get(
            f"{self.base_url}/api/locations/hierarchy",
            headers=self._headers(),
        )
        r.raise_for_status()

        data = r.json()
        print("📍 Locations:\n", json.dumps(data, indent=2))

        return data

    def get_workflows(self):
        r = requests.get(
            f"{self.base_url}/api/workflows",
            headers=self._headers(),
        )
        r.raise_for_status()

        data = r.json()
        print("⚙️ Workflows:\n", json.dumps(data, indent=2))

        return data

    def get_departments(self):
        r = requests.get(
            f"{self.base_url}/api/departments",
            headers=self._headers(),
        )
        r.raise_for_status()

        data = r.json()
        print("🏢 Departments:\n", json.dumps(data, indent=2))

        return data

    # -----------------------------
    # ATTACHMENT UPLOAD
    # -----------------------------

    def create_attachment(self, image_data):
        """
        Upload an image attachment and return the attachment ID.

        Args:
            image_data: Either raw bytes or a base64 encoded string
        """
        url = f"{self.base_url}/api/compose/namespace/{self.namespaceid}/module/{self.moduleid}/record/attachment"

        if isinstance(image_data, bytes):
            img_bytes = image_data
        else:
            image_base64 = image_data
            if "," in image_base64:
                image_base64 = image_base64.split(",", 1)[1]

            padding = 4 - len(image_base64) % 4
            if padding != 4:
                image_base64 += "=" * padding

            img_bytes = base64.b64decode(image_base64)

        print(f"📷 Image size: {len(img_bytes)} bytes")

        headers = {"Authorization": f"Bearer {self.token}"}
        body = {"fieldName": "Attachments"}
        files = {"upload": ("image.jpg", img_bytes, "image/jpeg")}

        r = requests.post(url, headers=headers, data=body, files=files)

        print(f"📤 Attachment upload response: status={r.status_code}, content-type={r.headers.get('content-type', 'unknown')}")
        print(f"📤 Response text (first 500 chars): {r.text[:500] if r.text else 'EMPTY'}")

        r.raise_for_status()

        if not r.text or not r.text.strip():
            print("⚠️ Empty response from attachment upload - returning placeholder ID")
            return "attachment_uploaded_no_id"

        try:
            data = r.json()
            attachment_id = data.get("response", {}).get("attachmentID", "")
            if not attachment_id:
                attachment_id = data.get("attachmentID", "") or data.get("id", "") or str(data)
            print(f"📎 Attachment uploaded: {attachment_id}")
            return attachment_id
        except Exception as e:
            print(f"⚠️ Could not parse attachment response as JSON: {e}")
            print(f"⚠️ Raw response: {r.text[:200]}")
            return "attachment_uploaded_parse_error"

    def attach_file_to_incident(self, incident_id: str, file_path: str):
        """
        Attach a local file to an existing incident in AutoMax.

        Args:
            incident_id: The record ID of the existing incident
            file_path: Absolute path to the file to attach
        """
        url = f"{self.base_url}/api/compose/namespace/{self.namespaceid}/module/{self.moduleid}/record/{incident_id}/attachment"

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_name = os.path.basename(file_path)
        headers = {"Authorization": f"Bearer {self.token}"}
        body = {"fieldName": "Attachments"}

        with open(file_path, "rb") as f:
            file_bytes = f.read()

        # Detect MIME type by extension
        ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else "bin"
        mime_map = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
            "gif": "image/gif", "pdf": "application/pdf",
            "txt": "text/plain", "csv": "text/csv",
        }
        mime_type = mime_map.get(ext, "application/octet-stream")

        files = {"upload": (file_name, file_bytes, mime_type)}

        r = requests.post(url, headers=headers, data=body, files=files)

        print(f"📎 Attach file to incident {incident_id}: status={r.status_code}")
        r.raise_for_status()

        if not r.text or not r.text.strip():
            return {"status": "attached", "incident_id": incident_id}

        try:
            return r.json()
        except Exception:
            return {"status": "attached", "incident_id": incident_id, "raw": r.text[:200]}

    # -----------------------------
    # INCIDENT CREATION
    # -----------------------------

    def create_incident(
        self,
        title: str,
        description: str,
        classification: str,
        location: str,
        workflow: str,
        priority: str = "LOW",
        severity: str = "LOW",
        reporter_name: str = "",
        reporter_id: str = "45",
        coordinates: str = "",
        channel: str = "Chatbot",
        national_id: str = "45",
    ):
        url = f"{self.base_url}/api/compose/namespace/{self.namespaceid}/module/{self.moduleid}/record/"

        body = {
            "meta": {},
            "records": [],
            "values": [
                {"name": "Channel", "value": channel},
                {"name": "Criticality", "value": priority},
                {"name": "Severity", "value": severity},
                {"name": "Title", "value": title},
                {"name": "Caller_name", "value": reporter_name},
                {"name": "Last_call_date", "value": datetime.utcnow().isoformat() + "Z"},
                {"name": "National_ID", "value": reporter_id or national_id},
                {"name": "Mobile_number", "value": ""},
                {"name": "Classification", "value": classification},
                {"name": "Incident_reason", "value": ""},
                {"name": "Incident_Description", "value": description},
                {"name": "Map", "value": coordinates},
                {"name": "Primary_Location", "value": location},
                {"name": "Workflow", "value": workflow},
                {"name": "District", "value": "Nanded"},
                {"name": "Street", "value": ""},
                {"name": "Status", "value": ""},
                {"name": "Assigned_To", "value": "425635139776282625"},
                {
                    "name": "Comments",
                    "value": json.dumps({
                        "created": datetime.utcnow().isoformat() + "Z",
                        "comment": description,
                        "author": "mcp@system",
                        "name": reporter_name or "MCP Bot",
                    }),
                },
            ],
        }

        r = requests.post(url, json=body, headers=self._headers())
        r.raise_for_status()

        data = r.json()
        print("🚨 Incident created:\n", json.dumps(data, indent=2))

        return data
