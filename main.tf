# Make sure to create state bucket beforehand
terraform {
  required_version = ">= 1.0"
  backend "s3" {
    bucket  = "mlops-course-project"
    key     = "mlops-course-project.tfstate"
    region  = "eu-central-1"
    encrypt = true
  }
}

provider "aws" {
  region = var.aws_region
}

data "aws_caller_identity" "current_identity" {}

resource "random_string" "random_suffix" {
  length  = 8
  special = false
  upper   = false
}

# model bucket
module "s3_bucket" {
  source      = "./modules/s3"
  bucket_name = "${var.model_bucket}-${random_string.random_suffix.result}"
}

# image registry
module "ecr_image" {
  source        = "./modules/ecr"
  ecr_repo_name = "${var.ecr_repo_name}_${random_string.random_suffix.result}"
}
