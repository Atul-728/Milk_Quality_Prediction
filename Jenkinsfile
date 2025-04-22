pipeline {
    agent any

    environment {
        IMAGE_NAME = 'milk_quality_predictor'
        DOCKER_HUB_REPO = '12211415/milk_quality_predictor'
        DOCKER_TAG = "${BUILD_NUMBER}"  // Using build number for unique tagging
    }

    stages {
        stage('Checkout') {
            steps {
                // Checkout the code from the repository
                checkout scm
            }
        }

        stage('Build Image') {
            steps {
                script {
                    // Build the Docker image with the build number as a tag
                    docker.build("${DOCKER_HUB_REPO}:${DOCKER_TAG}")
                }
            }
        }

        stage('Run Tests') {
            steps {
                // Run unit tests to ensure the project is working correctly
                sh 'python -m unittest discover tests'
            }
        }

        stage('Push to Docker Hub') {
            steps {
                withDockerRegistry([credentialsId: 'dockerhub-credentials', url: 'https://index.docker.io/v1/']) {
                    script {
                        // Push the image with the tag (e.g., build number or latest)
                        docker.image("${DOCKER_HUB_REPO}:${DOCKER_TAG}").push()
                        docker.image("${DOCKER_HUB_REPO}:${DOCKER_TAG}").push('latest')  // Optionally, push latest as well
                    }
                }
            }
        }
    }

    post {
        always {
            echo 'Pipeline finished.'
        }
        failure {
            echo 'Pipeline failed.'
        }
    }
}
