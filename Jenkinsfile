pipeline {
    agent any

    environment {
        // Set the GitHub repository URL
        GIT_URL = 'https://github.com/Atul-728/Milk_Quality_Prediction.git'
        DOCKER_IMAGE = 'milk-quality-app:latest'
    }

    stages {
        stage('Checkout') {
            steps {
                // Checkout the Git repository
                git url: "${GIT_URL}"
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    // Build the Docker image from the Dockerfile
                    docker.build(DOCKER_IMAGE)
                }
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    // Run tests inside the Docker container (ensure you have pytest in your requirements)
                    docker.image(DOCKER_IMAGE).inside {
                        sh 'pytest tests/'  // Adjust the path to your test folder if needed
                    }
                }
            }
        }

        stage('Deploy') {
            steps {
                script {
                    // Run the container with the app (expose to port 5000)
                    docker.image(DOCKER_IMAGE).run('-p 5000:5000')
                }
            }
        }
    }

    post {
        always {
            // Clean up the Docker image after the build
            sh 'docker system prune -f'
        }
    }
}
