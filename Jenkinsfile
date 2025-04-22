pipeline {
    agent any
    environment {
        DOCKER_HUB_USER = '12211415'
        REPO_NAME = 'milk_quality_prediction'
        DOCKER_IMAGE = "${DOCKER_HUB_USER}/${REPO_NAME}"
        DOCKER_TAG = 'latest'
    }
    stages {
        stage('Checkout') {
            steps { git branch: 'main', url: 'https://github.com/Atul-728/Milk_Quality_Prediction.git' }
        }
        stage('Build Image') {
            steps { script { docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}") } }
        }
        stage('Run Tests') {
            steps { bat 'python -m pytest tests\\' }
        }
        stage('Push to Docker Hub') {
            steps {
                script {
                    docker.withRegistry('https://registry.hub.docker.com', 'dockerhub-credentials') {
                        docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}").push()
                    }
                }
            }
        }
    }
}
