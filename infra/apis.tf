# Enable ONLY the APIs this sandbox needs. Every other GCP API stays off in the
# project, which is a core part of the isolation requirement.
locals {
  required_apis = [
    "compute.googleapis.com", # the VM, VPC, NAT, firewall
    "storage.googleapis.com", # the GCS bucket
    "iap.googleapis.com",     # browser-SSH / tunnel without a public IP
    "iam.googleapis.com",     # service accounts + impersonation
  ]
}

resource "google_project_service" "enabled" {
  for_each = toset(local.required_apis)

  service = each.value

  # Leave APIs enabled if the config is destroyed — disabling shared APIs is
  # disruptive and not what `terraform destroy` of a VM should do.
  disable_on_destroy         = false
  disable_dependent_services = false
}
