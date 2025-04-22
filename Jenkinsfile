pipeline {
    agent any
    
    environment {
        // Customized for your details
        DOCKER_HUB_USER = '12211415'  // Your Docker Hub username
        REPO_NAME = 'milk-quality-prediction'  // Docker image name
        DOCKER_IMAGE = "${DOCKER_HUB_USER}/${REPO_NAME}"
        DOCKER_TAG = 'latest'
        GIT_REPO = 'https://github.com/Atul-728/Milk_Quality_Prediction.git'
    }
    
    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: "${GIT_REPO}"  // Change branch if needed
            }
        }
        
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").inside {
                        sh 'python -m pytest tests/'  // Adjust test command
                    }
                }
            }
        }

        stage('Push to Docker Hub') {
            steps {
                script {
                    // Using 'github-credentials' for Docker Hub auth
                    docker.withRegistry('https://registry.hub.docker.com', 'github-credentials') {
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push()
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    sh """
                        docker stop milk-app || true
                        docker rm milk-app || true
                        docker run -d --name milk-app -p 5000:5000 ${DOCKER_IMAGE}:${DOCKER_TAG}
                    """
                }
            }
        }
    }
}