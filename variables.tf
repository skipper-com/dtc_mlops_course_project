variable "aws_region" {
  description = "Main region for all AWS resources"
  default     = "eu-central-1"
}

variable "project_id" {
  description = "project_id"
  default     = "123"
}

variable "model_bucket" {
  description = "s3_bucket"
  default     = "mlops-course-project"
}

variable "docker_image_local_path" {
  description = "Docker image path in ECR"
  default     = "mlops-course-project"
}

variable "ecr_repo_name" {
  description = "ECR repository name"
  default     = "mlops-course-project"
}
