from google.cloud import storage
import os
from datetime import datetime

print("=" * 70)
print("☁️  UPLOADING MODEL TO GOOGLE CLOUD STORAGE")
print("=" * 70)

# ============================================================================
# CONFIGURATION - UPDATE THESE VALUES
# ============================================================================
PROJECT_ID = "carbonsense-project"  # 🔴 CHANGE THIS TO YOUR PROJECT ID
BUCKET_NAME = f"carbonsense-data-hruthik123"  # 🔴 CHANGE THIS TO YOUR BUCKET NAME
MODEL_VERSION = "v1"

print(f"\n🔧 Configuration:")
print(f"   Project ID:    {PROJECT_ID}")
print(f"   Bucket Name:    {BUCKET_NAME}")
print(f"   Model Version:  {MODEL_VERSION}")

# ============================================================================
# INITIALIZE STORAGE CLIENT
# ============================================================================
print(f"\n🔐 Connecting to Google Cloud Storage...")
try:
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    print(f"✅ Connected successfully!")
except Exception as e:
    print(f"❌ Error connecting to Cloud Storage: {e}")
    print(f"   Make sure:")
    print(f"   1. GOOGLE_APPLICATION_CREDENTIALS is set")
    print(f"   2. Project ID is correct")
    exit(1)

# ============================================================================
# FILES TO UPLOAD
# ============================================================================
files_to_upload = [
    ('model_artifacts/carbon_model.pkl', 'model'),
    ('model_artifacts/metrics.json', 'metadata'),
    ('label_encoder.pkl', 'preprocessor'),
    ('scaler.pkl', 'preprocessor')
]

print(f"\n📋 Files to upload:")
for local_file, file_type in files_to_upload:
    if os.path.exists(local_file):
        file_size_kb = os.path.getsize(local_file) / 1024
        print(f"   ✅ {local_file:40s} ({file_size_kb:6.1f} KB)")
    else:
        print(f"   ❌ {local_file:40s} (NOT FOUND!)")

# ============================================================================
# UPLOAD EACH FILE
# ============================================================================
print(f"\n📤 Uploading files...")

for local_file, file_type in files_to_upload:
    if not os.path.exists(local_file):
        print(f"⚠️  Skipping {local_file} (not found)")
        continue
    
    try:
        # Create cloud path: models/v1/filename
        cloud_path = f"models/{MODEL_VERSION}/{os.path.basename(local_file)}"
        
        print(f"\n   Uploading: {local_file}")
        blob = bucket.blob(cloud_path)
        blob.upload_from_filename(local_file)
        
        # Get file size
        file_size_mb = blob.size / (1024 * 1024)
        print(f"   ✅ Success! ({file_size_mb:.2f} MB)")
        print(f"   📍 Location: gs://{BUCKET_NAME}/{cloud_path}")
        
    except Exception as e:
        print(f"   ❌ Error uploading {local_file}: {e}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ UPLOAD COMPLETE!")
print("=" * 70)

print(f"\n🔗 Your model files are now in Cloud Storage at:")
print(f"   gs://{BUCKET_NAME}/models/{MODEL_VERSION}/")

print(f"\n📋 Files uploaded:")
print(f"   ✅ carbon_model.pkl")
print(f"   ✅ metrics.json")
print(f"   ✅ label_encoder.pkl")
print(f"   ✅ scaler.pkl")

print(f"\n🚀 NEXT STEP:")
print(f"   1. Run: python package_for_deployment.py")
print(f"   2. Then deploy to Vertex AI")

print("\n" + "=" * 70)