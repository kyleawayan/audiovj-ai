###############################################################################
# runner-sa : the VM's identity.
# Deliberately has ZERO project-level role bindings. Its only grant is
# objectAdmin on the one bucket (in storage.tf). Combined with the narrowed
# instance OAuth scopes (compute.tf), the VM cannot reach any other GCP API —
# this is the heart of the isolation requirement.
###############################################################################
resource "google_service_account" "runner" {
  account_id   = "audiovj-runner"
  display_name = "AudioVJ GPU runner (bucket-only)"

  depends_on = [google_project_service.enabled]
}

###############################################################################
# transfer-sa : a bucket-only identity you IMPERSONATE from your laptop for
# keyless uploads/downloads (no JSON key files). You `gcloud auth login` as
# yourself, then `--impersonate-service-account=transfer-sa@...`.
###############################################################################
resource "google_service_account" "transfer" {
  account_id   = "audiovj-transfer"
  display_name = "AudioVJ bucket transfer (impersonation target)"

  depends_on = [google_project_service.enabled]
}

# Let your account mint short-lived tokens for transfer-sa (keyless impersonation).
resource "google_service_account_iam_member" "user_can_impersonate_transfer" {
  service_account_id = google_service_account.transfer.name
  role               = "roles/iam.serviceAccountTokenCreator"
  member             = "user:${var.iap_user}"
}

###############################################################################
# Your access to reach the VM via browser-SSH over IAP (no local SSH client).
# Note: if you are project Owner these are redundant for you, but they make the
# setup work for a non-owner identity too and document the minimum needed.
###############################################################################
resource "google_project_iam_member" "user_iap_tunnel" {
  project = var.project_id
  role    = "roles/iap.tunnelResourceAccessor"
  member  = "user:${var.iap_user}"
}

resource "google_project_iam_member" "user_os_login" {
  project = var.project_id
  role    = "roles/compute.osLogin"
  member  = "user:${var.iap_user}"
}
