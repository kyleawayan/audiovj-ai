# Custom VPC (no default network / no auto subnets) so nothing is implicitly open.
resource "google_compute_network" "vpc" {
  name                    = "audiovj-vpc"
  auto_create_subnetworks = false

  depends_on = [google_project_service.enabled]
}

resource "google_compute_subnetwork" "subnet" {
  name          = "audiovj-subnet"
  ip_cidr_range = "10.10.0.0/24"
  region        = var.region
  network       = google_compute_network.vpc.id

  # Lets the no-external-IP VM reach Google APIs (GCS, logging) directly.
  private_ip_google_access = true
}

# Ingress: only the IAP TCP-forwarding range, only tcp:22, only to the VM tag.
# This is what makes the console "SSH-in-browser" button work with no public IP.
resource "google_compute_firewall" "iap_ssh" {
  count = var.enable_iap ? 1 : 0

  name      = "allow-iap-ssh"
  network   = google_compute_network.vpc.name
  direction = "INGRESS"
  priority  = 1000

  allow {
    protocol = "tcp"
    ports    = ["22"]
  }

  source_ranges = ["35.235.240.0/20"] # Google IAP TCP forwarding range
  target_tags   = ["audiovj-runner"]
}

# Explicit deny-all ingress baseline (defense in depth; VPC already has no other
# allow rules, but this makes the intent unmistakable and logs blocked attempts).
resource "google_compute_firewall" "deny_all_ingress" {
  name      = "deny-all-ingress"
  network   = google_compute_network.vpc.name
  direction = "INGRESS"
  priority  = 65534

  deny {
    protocol = "all"
  }

  source_ranges = ["0.0.0.0/0"]
}

###############################################################################
# Outbound internet for a VM with no external IP — via Cloud NAT.
# Needed for: Anthropic API (Claude Code + WebSearch), PyPI (CUDA torch), apt.
# Google APIs (GCS) go via Private Google Access and don't traverse NAT.
###############################################################################
resource "google_compute_router" "router" {
  name    = "audiovj-router"
  region  = var.region
  network = google_compute_network.vpc.id
}

resource "google_compute_router_nat" "nat" {
  name                               = "audiovj-nat"
  router                             = google_compute_router.router.name
  region                             = var.region
  nat_ip_allocate_option             = "AUTO_ONLY"
  source_subnetwork_ip_ranges_to_nat = "ALL_SUBNETWORKS_ALL_IP_RANGES"

  log_config {
    enable = true
    filter = "ERRORS_ONLY"
  }
}

###############################################################################
# Optional hardened egress (harden_egress=true): deny all egress, then allow
# only Google APIs (for Private Google Access / GCS) + a configurable CIDR set.
# This is the real network-level guard against token exfil when sandbox is off.
###############################################################################
resource "google_compute_firewall" "egress_deny_all" {
  count = var.harden_egress ? 1 : 0

  name      = "egress-deny-all"
  network   = google_compute_network.vpc.name
  direction = "EGRESS"
  priority  = 65500

  deny {
    protocol = "all"
  }

  destination_ranges = ["0.0.0.0/0"]
}

resource "google_compute_firewall" "egress_allow_google_apis" {
  count = var.harden_egress ? 1 : 0

  name      = "egress-allow-google-apis"
  network   = google_compute_network.vpc.name
  direction = "EGRESS"
  priority  = 1000

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }

  # restricted.googleapis.com VIP — covers GCS/logging via Private Google Access.
  destination_ranges = ["199.36.153.4/30"]
}

resource "google_compute_firewall" "egress_allow_extra" {
  count = var.harden_egress && length(var.egress_allow_cidrs) > 0 ? 1 : 0

  name      = "egress-allow-extra"
  network   = google_compute_network.vpc.name
  direction = "EGRESS"
  priority  = 1001

  allow {
    protocol = "tcp"
    ports    = ["443"]
  }

  destination_ranges = var.egress_allow_cidrs
}
