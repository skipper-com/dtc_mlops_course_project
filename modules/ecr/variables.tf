variable "ecr_repo_name" {
  type        = string
  description = "ECR repo name"
}

variable "ecr_image_tag" {
  type        = string
  description = "ECR repo name"
  default     = "latest"
}

variable "region" {
  type        = string
  description = "region"
  default     = "eu-region-1"
}
