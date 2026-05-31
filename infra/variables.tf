###############################################################################
# Core / project
###############################################################################

variable "project_id" {
  type        = string
  description = "ID of the (already-created) GCP project to deploy into. Create the project and link billing yourself; Terraform only manages resources within it."
}

variable "region" {
  type        = string
  description = "GCP region. Kept in the US because Claude Code WebSearch is US-served."
  default     = "us-central1"
}

variable "zone" {
  type        = string
  description = "Zone for the GPU VM. Must offer the chosen GPU (us-central1-a has T4)."
  default     = "us-central1-a"
}

###############################################################################
# Machine / GPU  (workload is data-pipeline-bound, not GPU-bound:
# RAM + fast local disk matter more than the accelerator)
###############################################################################

variable "machine_type" {
  type        = string
  description = "VM machine type. n1-highmem-8 = 8 vCPU / 52GB RAM, so the OS page cache holds most of the ~47GB feature set after the first epoch."
  default     = "n1-highmem-8"
}

variable "gpu_type" {
  type        = string
  description = "Accelerator type. T4 is cheap and keeps up since the GPU is not the bottleneck. Use nvidia-l4 (with an n1->g2 machine_type change) only for bigger/stateful models."
  default     = "nvidia-tesla-t4"
}

variable "gpu_count" {
  type        = number
  description = "Number of GPUs (0 disables the accelerator entirely)."
  default     = 1
}

variable "local_ssd_count" {
  type        = number
  description = "375GB local NVMe SSD scratch disks. 1 holds the ~150GB extracted archive. >1 requires RAID (see README)."
  default     = 1
}

variable "boot_disk_gb" {
  type        = number
  description = "Boot disk size (Deep Learning VM image + CUDA + tooling)."
  default     = 100
}

variable "image_project" {
  type        = string
  description = "Source project for the boot image. Deep Learning VM images ship NVIDIA drivers + CUDA."
  default     = "deeplearning-platform-release"
}

variable "image_family" {
  type        = string
  description = "Boot image family. CUDA base on Ubuntu 22.04 with the NVIDIA driver preinstalled. List options: gcloud compute images list --project deeplearning-platform-release --filter='family~common-cu'"
  default     = "common-cu129-ubuntu-2204-nvidia-580"
}

variable "use_spot" {
  type        = bool
  description = "Spot/preemptible VM (~3x cheaper, can be reclaimed anytime). Default off for a reliable unattended loop."
  default     = false
}

###############################################################################
# Storage
###############################################################################

variable "bucket_name" {
  type        = string
  description = "Globally-unique name for the private GCS bucket (durable archive + transfer channel)."
}

variable "bucket_force_destroy" {
  type        = bool
  description = "Allow `terraform destroy` to delete the bucket even if it still holds objects. Off by default so you don't lose the archive."
  default     = false
}

variable "archive_object" {
  type        = string
  description = "Name of the data tar in the bucket. The startup script stream-extracts gs://<bucket>/<archive_object> onto the local SSD."
  default     = "archive.tar"
}

variable "repo_bundle_object" {
  type        = string
  description = "Name of the git bundle in the bucket. setup.sh clones the repo from gs://<bucket>/<repo_bundle_object> (no GitHub login on the VM)."
  default     = "repo.bundle"
}

variable "data_disk_gb" {
  type        = number
  description = "Optional persistent SSD data disk (survives stop/start). 0 = none (local SSD + bucket are used instead). Costs ~$0.17/GB-mo while it exists."
  default     = 0
}

###############################################################################
# Access / isolation
###############################################################################

variable "assign_external_ip" {
  type        = bool
  description = "Give the VM a public IP. Default off: connect via the console browser-SSH button (IAP). Cloud NAT provides outbound internet regardless."
  default     = false
}

variable "enable_iap" {
  type        = bool
  description = "Open tcp:22 to the IAP range so the console browser-SSH button / `--tunnel-through-iap` works without a public IP."
  default     = true
}

variable "iap_user" {
  type        = string
  description = "Your Google account email. Gets IAP-tunnel + OS Login (to browser-SSH in) and token-creator on the transfer SA (keyless bucket-only uploads)."
}

variable "harden_egress" {
  type        = bool
  description = "Replace open egress with an allow-list (Google APIs + a configurable CIDR set). Strong compensating control when Claude Code sandbox is off, but needs maintenance as CDN IPs move. Off by default."
  default     = false
}

variable "egress_allow_cidrs" {
  type        = list(string)
  description = "Extra egress CIDRs allowed when harden_egress=true (e.g. Anthropic API + PyPI ranges). Google API access is preserved separately via Private Google Access."
  default     = []
}

###############################################################################
# Guardrails
###############################################################################

variable "idle_shutdown_minutes" {
  type        = number
  description = "Shut the VM down after this many minutes of low CPU+GPU utilization. 0 disables."
  default     = 60
}

variable "enable_budget" {
  type        = bool
  description = "Create a billing budget + email alerts. Requires billing-account-level permission for the Terraform identity."
  default     = false
}

variable "billing_account" {
  type        = string
  description = "Billing account ID (e.g. 0X0X0X-0X0X0X-0X0X0X) for the budget. Only used when enable_budget=true."
  default     = ""
}

variable "budget_amount" {
  type        = number
  description = "Budget amount in USD."
  default     = 100
}

variable "budget_notify_email" {
  type        = string
  description = "Email for budget alerts (typically your own). Only used when enable_budget=true."
  default     = ""
}
