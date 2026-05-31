# Optional billing budget + alerts. Requires the Terraform identity to have
# billing.budgets.* on the billing account (often broader than project Owner),
# so it's off by default. Enable by setting enable_budget=true and billing_account.
#
# Alerts go to the billing account's IAM recipients (Billing Admins/Owners) by
# default — this avoids enabling the Monitoring API just to send an email, which
# keeps the project's API surface minimal (the isolation goal).
resource "google_billing_budget" "budget" {
  count = var.enable_budget ? 1 : 0

  billing_account = var.billing_account
  display_name    = "AudioVJ training sandbox budget"

  budget_filter {
    projects = ["projects/${var.project_id}"]
  }

  amount {
    specified_amount {
      currency_code = "USD"
      units         = tostring(var.budget_amount)
    }
  }

  threshold_rules {
    threshold_percent = 0.5
  }
  threshold_rules {
    threshold_percent = 0.9
  }
  threshold_rules {
    threshold_percent = 1.0
  }

  # No all_updates_rule block: threshold alerts then default to the billing
  # account's IAM recipients (Billing Admins/Users). Avoids a Monitoring channel
  # or Pub/Sub topic, keeping the project's API surface minimal.
}
