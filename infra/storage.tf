# Private bucket: the durable archive + the only channel for code-in / results-out.
resource "google_storage_bucket" "data" {
  name     = var.bucket_name
  location = var.region

  # No object ACLs — access is governed purely by the IAM bindings below.
  uniform_bucket_level_access = true

  # Never publicly accessible.
  public_access_prevention = "enforced"

  versioning {
    enabled = true
  }

  force_destroy = var.bucket_force_destroy

  depends_on = [google_project_service.enabled]
}

# The VM (runner-sa) can read/write THIS bucket and nothing else.
resource "google_storage_bucket_iam_member" "runner_object_admin" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.runner.email}"
}

# The keyless transfer identity (impersonated from your laptop) — bucket-only.
resource "google_storage_bucket_iam_member" "transfer_object_admin" {
  bucket = google_storage_bucket.data.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.transfer.email}"
}
