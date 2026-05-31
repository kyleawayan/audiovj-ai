# Optional persistent data disk (only when data_disk_gb > 0). By default we use
# local NVMe SSD + the bucket instead, so there's no idle disk charge.
resource "google_compute_disk" "data" {
  count = var.data_disk_gb > 0 ? 1 : 0

  name = "audiovj-data"
  type = "pd-ssd"
  zone = var.zone
  size = var.data_disk_gb

  depends_on = [google_project_service.enabled]
}

resource "google_compute_instance" "runner" {
  name         = "audiovj-runner"
  machine_type = var.machine_type
  zone         = var.zone

  # Tag drives the IAP SSH firewall rule.
  tags = ["audiovj-runner"]

  boot_disk {
    initialize_params {
      image = "projects/${var.image_project}/global/images/family/${var.image_family}"
      size  = var.boot_disk_gb
      type  = "pd-balanced"
    }
  }

  # 375GB local NVMe SSD scratch disk(s) — fast random reads for the mmap'd
  # feature set. Ephemeral: wiped on stop/terminate (re-extracted from bucket).
  dynamic "scratch_disk" {
    for_each = range(var.local_ssd_count)
    content {
      interface = "NVME"
    }
  }

  dynamic "attached_disk" {
    for_each = var.data_disk_gb > 0 ? [1] : []
    content {
      source      = google_compute_disk.data[0].id
      device_name = "audiovj-data"
    }
  }

  dynamic "guest_accelerator" {
    for_each = var.gpu_count > 0 ? [1] : []
    content {
      type  = var.gpu_type
      count = var.gpu_count
    }
  }

  # GPU VMs cannot live-migrate, so TERMINATE on host maintenance is required.
  scheduling {
    on_host_maintenance = "TERMINATE"
    automatic_restart   = var.use_spot ? false : true
    preemptible         = var.use_spot
    provisioning_model  = var.use_spot ? "SPOT" : "STANDARD"
  }

  network_interface {
    network    = google_compute_network.vpc.id
    subnetwork = google_compute_subnetwork.subnet.id

    # Only attach a public IP if explicitly requested. Default: none (IAP only).
    dynamic "access_config" {
      for_each = var.assign_external_ip ? [1] : []
      content {}
    }
  }

  service_account {
    email = google_service_account.runner.email
    # Narrow OAuth scopes: even though runner-sa has no other roles, this caps
    # the metadata token to storage read/write + log writing.
    scopes = [
      "https://www.googleapis.com/auth/devstorage.read_write",
      "https://www.googleapis.com/auth/logging.write",
    ]
  }

  metadata = {
    # DLVM auto-installs the NVIDIA driver on first boot.
    install-nvidia-driver = "True"
    enable-oslogin        = "TRUE"

    # Consumed by startup.sh via the metadata server (avoids templatefile
    # escaping and keeps NO secrets in metadata — these are all non-sensitive).
    bucket-name           = var.bucket_name
    archive-object        = var.archive_object
    repo-bundle           = var.repo_bundle_object
    idle-shutdown-minutes = tostring(var.idle_shutdown_minutes)

    # Minimal Claude Code settings, deployed to new users via /etc/skel.
    claude-settings = file("${path.module}/claude-settings.json")
  }

  metadata_startup_script = file("${path.module}/startup.sh")

  # Allows changing GPU/scopes/etc. via `terraform apply` (stops, edits, starts).
  allow_stopping_for_update = true

  labels = {
    purpose = "audiovj-training"
    managed = "terraform"
  }

  depends_on = [
    google_project_service.enabled,
    google_compute_router_nat.nat,
  ]
}
