import os
from google.cloud import secretmanager

def get_secret(secret_id_env_var):
    """
    Retrieves a secret from Google Secret Manager using a resource ID 
    stored in an environment variable.
    """
    secret_resource_id = os.environ.get(secret_id_env_var)
    if not secret_resource_id:
        # Fallback to direct environment variable if resource ID is not provided
        # (Useful for local dev if not using Secret Manager)
        return os.environ.get(secret_id_env_var.replace("_ID", ""))

    try:
        client = secretmanager.SecretManagerServiceClient()
        response = client.access_secret_version(request={"name": secret_resource_id})
        return response.payload.data.decode("UTF-8")
    except Exception as e:
        print(f"Error retrieving secret {secret_id_env_var}: {e}")
        # Fallback to direct environment variable
        return os.environ.get(secret_id_env_var.replace("_ID", ""))
