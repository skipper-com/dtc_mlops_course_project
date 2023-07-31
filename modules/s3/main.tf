resource "aws_s3_bucket" "s3_bucket" {
  bucket        = var.bucket_name
  force_destroy = true
  depends_on    = [aws_s3_bucket_ownership_controls.s3_bucket_acl_ownership]
}

resource "aws_s3_bucket_ownership_controls" "s3_bucket_acl_ownership" {
  bucket = var.bucket_name
  rule {
    object_ownership = "ObjectWriter"
  }
}

output "name" {
  value = aws_s3_bucket.s3_bucket.bucket
}
