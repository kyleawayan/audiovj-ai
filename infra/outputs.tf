output "bucket_name" {
  description = "Private GCS bucket for archive + transfers."
  value       = google_storage_bucket.data.name
}

output "runner_service_account" {
  description = "Instance identity (bucket-only, no other GCP access)."
  value       = google_service_account.runner.email
}

output "transfer_service_account" {
  description = "Impersonate this for keyless, bucket-only uploads from your laptop."
  value       = google_service_account.transfer.email
}

output "instance_name" {
  value = google_compute_instance.runner.name
}

output "instance_zone" {
  value = google_compute_instance.runner.zone
}

output "external_ip" {
  description = "Public IP, only if assign_external_ip=true (otherwise empty — connect via IAP)."
  value       = try(google_compute_instance.runner.network_interface[0].access_config[0].nat_ip, "")
}

output "browser_ssh_url" {
  description = "Open this in a browser to SSH into the VM over IAP (no local SSH/keys)."
  value       = "https://ssh.cloud.google.com/v2/ssh/projects/${var.project_id}/zones/${var.zone}/instances/${google_compute_instance.runner.name}?useAdminProxy=true&troubleshoot=true"
}

output "ssh_via_iap_command" {
  description = "Alternative: SSH from a terminal that has gcloud (still no public IP)."
  value       = "gcloud compute ssh ${google_compute_instance.runner.name} --zone ${var.zone} --tunnel-through-iap --project ${var.project_id}"
}

output "upload_archive_command" {
  description = "Keyless, bucket-only upload of your data tar from your laptop."
  value       = "gcloud storage cp <your-archive>.tar gs://${google_storage_bucket.data.name}/${var.archive_object} --impersonate-service-account=${google_service_account.transfer.email}"
}

output "upload_repo_bundle_command" {
  description = "Build + upload the code bundle (run inside your local repo)."
  value       = "git bundle create /tmp/${var.repo_bundle_object} --all && gcloud storage cp /tmp/${var.repo_bundle_object} gs://${google_storage_bucket.data.name}/${var.repo_bundle_object} --impersonate-service-account=${google_service_account.transfer.email}"
}
