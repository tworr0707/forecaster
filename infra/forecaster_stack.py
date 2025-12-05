import os
from typing import Optional

import aws_cdk as cdk
from aws_cdk import (
    aws_ec2 as ec2,
    aws_iam as iam,
    aws_rds as rds,
    aws_s3 as s3,
)


class ForecasterStack(cdk.Stack):
    """Minimal CDK stack for VPC + Aurora Serverless v2 + EC2 GPU with Bedrock access."""

    def __init__(self, scope: cdk.App, construct_id: str, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Parameters / context defaults
        region = self.region
        vpc_cidr = self.node.try_get_context("vpc_cidr") or "10.42.0.0/16"
        min_acu = float(self.node.try_get_context("aurora_min_acu") or 0.5)
        max_acu = float(self.node.try_get_context("aurora_max_acu") or 4)
        db_name = self.node.try_get_context("db_name") or "forecaster"
        instance_type = self.node.try_get_context("ec2_instance_type") or "p5.2xlarge"
        key_name = self.node.try_get_context("ec2_key_name")  # optional
        plots_bucket_name = self.node.try_get_context("plots_bucket") or "forecaster-plots-dev"
        bedrock_models_arn = self.node.try_get_context("bedrock_model_arn") or "*"  # scope later

        # VPC with 2 private + 1 public subnet, 1 NAT
        vpc = ec2.Vpc(
            self,
            "ForecasterVpc",
            cidr=vpc_cidr,
            max_azs=2,
            nat_gateways=1,
            subnet_configuration=[
                ec2.SubnetConfiguration(name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24),
                ec2.SubnetConfiguration(name="app", subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS, cidr_mask=24),
                ec2.SubnetConfiguration(name="db", subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, cidr_mask=24),
            ],
        )

        # Security Groups
        ec2_sg = ec2.SecurityGroup(self, "Ec2SG", vpc=vpc, allow_all_outbound=True)
        db_sg = ec2.SecurityGroup(self, "DbSG", vpc=vpc, allow_all_outbound=False)
        db_sg.add_ingress_rule(peer=ec2_sg, connection=ec2.Port.tcp(5432), description="EC2 to Aurora")

        # Aurora Serverless v2 Postgres with pgvector parameter group
        parameter_group = rds.ParameterGroup(
            self,
            "AuroraParams",
            engine=rds.DatabaseClusterEngine.AURORA_POSTGRESQL,
            parameters={"shared_preload_libraries": "pgvector"},
        )

        cluster = rds.ServerlessCluster(
            self,
            "AuroraCluster",
            engine=rds.DatabaseClusterEngine.AURORA_POSTGRESQL,
            default_database_name=db_name,
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_ISOLATED, one_per_az=True),
            security_groups=[db_sg],
            scaling=rds.ServerlessScalingOptions(min_capacity=min_acu, max_capacity=max_acu),
            parameter_group=parameter_group,
            enable_data_api=True,
            credentials=rds.Credentials.from_generated_secret("auroraadmin"),
        )

        # S3 bucket for plots (private, encrypted)
        plots_bucket = s3.Bucket(
            self,
            "PlotsBucket",
            bucket_name=plots_bucket_name if plots_bucket_name != "" else None,
            encryption=s3.BucketEncryption.S3_MANAGED,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            enforce_ssl=True,
            versioned=False,
            removal_policy=cdk.RemovalPolicy.RETAIN,
        )

        # IAM role for EC2
        ec2_role = iam.Role(
            self,
            "Ec2Role",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[iam.ManagedPolicy.from_aws_managed_policy_name("AmazonSSMManagedInstanceCore")],
        )

        # Allow Bedrock invoke
        ec2_role.add_to_policy(
            iam.PolicyStatement(
                actions=["bedrock:InvokeModel", "bedrock:InvokeModelWithResponseStream"],
                resources=[bedrock_models_arn],
            )
        )
        # Allow S3 bucket access
        plots_bucket.grant_read_write(ec2_role)
        # Allow reading DB secret
        cluster.secret.grant_read(ec2_role)

        # EC2 GPU instance
        ami = ec2.MachineImage.latest_amazon_linux2023(
            cpu_type=ec2.AmazonLinuxCpuType.X86_64,
            edition=ec2.AmazonLinuxEdition.STANDARD,
            virtualization=ec2.AmazonLinuxVirt.HVM,
            storage=ec2.AmazonLinuxStorage.GENERAL_PURPOSE,
        )

        ec2_instance = ec2.Instance(
            self,
            "ForecasterGpu",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            instance_type=ec2.InstanceType(instance_type),
            machine_image=ami,
            role=ec2_role,
            security_group=ec2_sg,
            key_name=key_name,
        )

        # Pass DB secret ARN and bucket/env into instance via user data env exports
        user_data_lines = [
            f"echo 'export AURORA_SECRET_ARN={cluster.secret.secret_arn}' >> /etc/profile.d/forecaster.sh",
            f"echo 'export PLOTS_BUCKET={plots_bucket.bucket_name}' >> /etc/profile.d/forecaster.sh",
            f"echo 'export AWS_REGION={region}' >> /etc/profile.d/forecaster.sh",
        ]
        ec2_instance.add_user_data("\n".join(user_data_lines))

        # VPC Endpoints (optional but keeps traffic inside VPC)
        vpc.add_gateway_endpoint(
            "S3Endpoint",
            service=ec2.GatewayVpcEndpointAwsService.S3,
            subnets=[ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS)],
        )
        vpc.add_interface_endpoint(
            "BedrockEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService("bedrock-runtime"),  # runtime endpoint name
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            security_groups=[ec2_sg],
        )
        vpc.add_interface_endpoint(
            "SecretsEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SECRETS_MANAGER,
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            security_groups=[ec2_sg],
        )
        vpc.add_interface_endpoint(
            "SsmEndpoint",
            service=ec2.InterfaceVpcEndpointAwsService.SSM,
            subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PRIVATE_WITH_EGRESS),
            security_groups=[ec2_sg],
        )

        # Outputs
        cdk.CfnOutput(self, "AuroraEndpoint", value=cluster.cluster_endpoint.hostname)
        cdk.CfnOutput(self, "AuroraSecretArn", value=cluster.secret.secret_arn)
        cdk.CfnOutput(self, "PlotsBucketName", value=plots_bucket.bucket_name)
        cdk.CfnOutput(self, "InstanceId", value=ec2_instance.instance_id)
        cdk.CfnOutput(self, "InstanceAz", value=ec2_instance.instance_availability_zone)
