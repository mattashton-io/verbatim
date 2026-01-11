import os
from google.cloud import secretmanager

def get_secret():
    """
    Retrieves a secret from Google Secret Manager using a resource ID 
    stored in an environment variable.
    """
    # secret_resource_id = os.environ.get(secret_id_env_var)
    # if not secret_resource_id:
    #     # Fallback to direct environment variable if resource ID is not provided
    #     # (Useful for local dev if not using Secret Manager)
    #     return os.environ.get(secret_id_env_var.replace("_ID", ""))

    try:
        # Create the Secret Manager client.
        client = secretmanager.SecretManagerServiceClient()

        # Build the resource name of the secret version.
        name = "projects/396631018769/secrets/verbatim-gemini/versions/latest"

        # Access the secret version.
        response = client.access_secret_version(request={"name": name})

        # Extract the payload.
        secret_string = response.payload.data.decode("UTF-8")
        return secret_string

    except Exception as e:
        print(f"Error retrieving secret {name}: {e}")

