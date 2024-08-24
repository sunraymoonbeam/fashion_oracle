# Table of contents
[Common Gcloud Commands](#top-k-gcloud-commands)
[Authentication & Configuration](#authentication--configuration)
[K8s Resource Management](#k8s-resource-management)
[Compute Instance Resource Management](#compute-instance-resource-management)
[App Engine](#app-engine)
[Differences Between Compute Instance, Kubernetes (K8s), and App Engine](#differences-between-compute-instance-kubernetes-k8s-and-app-engine)

[What We Did During MLOps Week](#what-we-did-during-mlops-week)
[First Attempt](#first-attempt)
[Second Attempt](#second-attempt)

[Questions and Answers](#questions-and-answers)
---


## Top K Gcloud commands
### Authentication / Configuration
---

* ```gcloud init```: Initializes the gcloud CLI and sets up the configuration, allowing you to select an account and a project.

* ```gcloud auth login```: Authenticates a user account for user-based authentication.

* ```gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS```: Activates a service account using the specified key file for service-based authentication.

* ```gcloud auth list```: Lists all authenticated accounts and indicates which one is currently active.

* ```gcloud config list```: Lists the current configuration, including the active account and project.

* ```gcloud config set account ACCOUNT```: Sets the active account.

* ```gcloud config set project PROJECT_ID```: Sets the active project.

* ```gcloud projects list```: Lists all projects.


### K8s Resource Management
---
* ```gcloud container clusters list```: Lists all Kubernetes clusters in the active project.

* ```gcloud container clusters get-credentials CLUSTER_NAME --zone=ZONE```: Fetches credentials for a Kubernetes cluster and configures kubectl.

* ```gcloud container clusters create CLUSTER_NAME --zone=ZONE```: Creates a new Kubernetes cluster.

* ```gcloud container clusters delete CLUSTER_NAME --zone=ZONE```: Deletes a Kubernetes cluster.

* ```gcloud container node-pools list --cluster=CLUSTER_NAME --zone=ZONE```: Lists all node pools in a Kubernetes cluster.


### Compute Instance Resource Management
---

* ```gcloud compute instances list```: Lists all Compute Engine instances in the active project.

* ```gcloud compute instances create INSTANCE_NAME --zone=ZONE```: Creates a new Compute Engine instance.

* ```gcloud compute instances delete INSTANCE_NAME --zone=ZONE```: Deletes a Compute Engine instance.

* ```gcloud compute disks list```: Lists all persistent disks in the active project.
---
## App Engine
* ```gcloud app deploy```: Deploys the current application to Google App Engine.

* ```gcloud app versions list```: Lists all versions of the App Engine application.

* ```gcloud app services set-traffic SERVICE --splits VERSION=TRAFFIC_PERCENTAGE```: Sets the traffic split for different versions of the App Engine service.

* ```gcloud app versions delete VERSION```: Deletes a specific version of the App Engine application.

* ```gcloud app describe```: Describes the App Engine application, including its configuration and status.


## Compute Instance vs K8s vs App Engine
https://cloud.google.com/kubernetes-engine?hl=en
https://cloud.google.com/appengine?hl=en
https://cloud.google.com/compute/docs/instances

#### Compute Instance (IaaS)
---

* **Infrastructure as a Service (IaaS)**: Provides virtual machines (VMs) that you can configure and manage.

* **Use Cases**: Custom applications, legacy applications, and workloads requiring full control over the operating system and environment.

* **Flexibility**: High flexibility and control over the environment, including the choice of OS, software, and configurations.

* **Management**: Requires manual management of the VMs, including scaling, updates, and maintenance.

#### Kubernetes k8s (CaaS)
---
* **Container Orchestration**: Manages containerized applications across a cluster of machines.

* **Use Cases**: Microservices, containerized applications, and applications requiring high scalability and resilience.

* **Flexibility**: Provides a high level of flexibility and control over containerized applications, including automated scaling, self-healing, and rolling updates.

* **Management**: Requires management of the Kubernetes cluster, including node management, networking, and security.

#### App Engine (PaaS)
---
* **Platform as a Service (PaaS)**: Provides a fully managed platform for building and deploying applications.'

* **Use Cases**: Web applications, mobile backends, and APIs.

* **Flexibility**: Limited flexibility compared to IaaS and Kubernetes, as it abstracts away the underlying infrastructure.

* **Management**: Minimal management required, as Google handles scaling, updates, and maintenance.

### Why Use Kubernetes Over App Engine
* **Flexibility and Control**: Kubernetes offers more control over the environment and application deployment compared to App Engine, which abstracts away much of the underlying infrastructure.

* **Portability**: Kubernetes is an open-source platform that can run on various cloud providers and on-premises environments, reducing vendor lock-in.

* **Microservices Architecture**: Kubernetes is well-suited for managing microservices architectures, providing features like service discovery, load balancing, and automated rollouts.

* **Scalability**: Kubernetes provides advanced scaling capabilities, including horizontal pod autoscaling and cluster autoscaling.

* **Custom Workloads**: Kubernetes can run a wide range of workloads, including stateful applications, batch processing, and machine learning workloads.

#### Vendor Lock-In
App Engine: As a fully managed PaaS, App Engine can lead to vendor lock-in due to its proprietary features and APIs. Migrating applications from App Engine to another platform may require significant changes to the application code and architecture.

Kubernetes: Being an open-source platform, Kubernetes reduces vendor lock-in by providing a consistent environment across different cloud providers and on-premises infrastructure. Applications deployed on Kubernetes can be more easily migrated between environments.

#### Summary
- Compute Instance (IaaS): Provides maximum control and flexibility but requires more management effort.

- Kubernetes (Container Orchestration): Offers a balance of control and automation, suitable for containerized applications and microservices.

- App Engine (PaaS): Provides a fully managed platform with minimal management effort but less flexibility and potential vendor lock-in.



## What We Did During MLOPS Week

1. ```gcloud init```

2. ```gcloud auth activate-service-account SERVICE_ACCOUNT --key-file=gcloud_config.json```

3. ```gcloud auth configure-docker asia-southeast1-docker.pkg.dev``` (for environments with docker)

4. ```docker push```

Explanation: ```gcloud init``` is used to specify the project or server and other configurations. You can then use ```gcloud auth login``` or ```gcloud auth activate-service-account``` to log in to the correct account linked to the project.

##  First Attempt

```gcloud init```

I was presented with an account already set up for me, **PROJECT-ID.svc.id.goog**, so I selected it.

```
khull kaniko --context $(pwd) \
--dockerfile $(pwd)/Dockerfile \
--destination ......
--cred-file $GOOGLE_APPLICATION_CREDENTIALS \
-v pvc-data:/pvc-data
```

Attempted to push with Kaniko using this account but encountered a 403 permission denied error.

## Second Attempt
```gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS```

Remembered from MLOps week that we authenticated with a service account. Assumed the service account had the necessary permissions, but still received a 403 permission denied error.

## Questions and answers
1. **What is a project?**

A project in Google Cloud is a container for all your Google Cloud resources. It provides:
* Resource Management: Organizes and manages resources like VMs, databases, and storage.

* Billing: Tracks usage and costs.

* Permissions: Manages access control through IAM (Identity and Access Management).

Each project has a unique project ID and can have multiple users and service accounts associated with it.

2. **What is a service account and how does it differ from a user account?**
### Service Account
* Purpose: Used by applications, VMs, or other services to authenticate and interact with Google Cloud APIs.

* Authentication: Typically uses a JSON key file.

* Permissions: Granted specific roles and permissions via IAM.

### User Account
* Purpose: Represents a human user who interacts with Google Cloud services.

* Authentication: Uses OAuth 2.0 for authentication (e.g., gcloud auth login).

* Permissions: Granted roles and permissions via IAM, similar to service accounts.


3. **Was the initial account PROJECT-ID.svc.id.goog a default user account?**
The account **PROJECT-ID.svc.id.goog** is not a typical user account. It appears to be a Google Cloud service identity, which is automatically managed by Google Cloud. These identities are used internally by Google Cloud services to manage resources and services. It is not intended for direct use by users or applications.

Upon further investigation, it seemed to be a workload identity, which is a binding of GOOGLE SERVICE ACCOUNTS TO KUBERNETES SERVICE ACCOUNTS...NEED TO LOOK INTO THIS
https://stackoverflow.com/questions/66985332/gke-workload-identity-pool-vs-workload-identity-pools-from-workload-identity-fed

https://medium.com/google-cloud/understanding-workload-identity-in-gke-2e622aaa7069

```
Workload Identity is a GKE Addon. Before going deeper, you have to know that, on Google Cloud Platform, you don't need to use service account key file because the service account are automatically loaded on every services (Compute Engine, App Engine, Cloud Run, Cloud Function, Cloud Build,...) and accessible through the metadata server . The Google Cloud client libraries automatically detect the environment and use the metadata server if present.

The problem with GKE is that you can run container on several different Compute Engine instances (the nodes) and your different service (K8S services) can have different level of authorization. If you rely on the Compute Engine service account (default behavior without the Workload identity addon), all the pods on the same instances use the same service account (and thus have the same permissions).

To solve that, the Workload Identity addon creates a proxy that intercept the metadata server calls and reply with the correct bind service account for this pods/service on GKE
```

### Key file for authentication
A key file is a JSON file that contains the credentials for a Google Cloud service account. This file is used to authenticate and authorize applications or services to interact with Google Cloud APIs on behalf of the service account.

Key Components of a Key File
* **Type**: Specifies the type of the key, usually "service_account".
* **Project ID**: The ID of the Google Cloud project associated with the service account.
* **Private Key ID**: A unique identifier for the private key.
* **Private Key**: The actual private key used for authentication.
* **Client Email**: The email address associated with the service account.
* **Client ID**: A unique identifier for the service account.
* **Auth URI**: The URI used for authentication.
* **Token URI**: The URI used to obtain OAuth 2.0 tokens.
* **Auth Provider X509 Cert URL**: The URL for the public X.509 certificate.
* **Client X509 Cert URL**: The URL for the service account's public X.509 certificate.