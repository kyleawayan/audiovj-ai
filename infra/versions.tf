terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
  }

  # Optional remote state. Uncomment and point at a bucket you control if you
  # want shared/durable state instead of local terraform.tfstate.
  # backend "gcs" {
  #   bucket = "your-tfstate-bucket"
  #   prefix = "audiovj-runner"
  # }
}

# Authenticates via your local gcloud Application Default Credentials:
#   gcloud auth application-default login
provider "google" {
  project = var.project_id
  region  = var.region
  zone    = var.zone
}
