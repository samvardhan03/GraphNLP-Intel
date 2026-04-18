# TODO: provision ECS / Cloud Run service + ElastiCache Redis + managed Neo4j (Aura)
terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}
