pipeline {
    agent any

    environment {
        IMAGE_NAME = 'milk_quality_predictor'
        DOCKER_HUB_REPO = '12211415/milk_quality_predictor'
    }

    stages {
        stage('Checkout') {
            steps {
                // This will use whatever branch Jenkins already checked out
                checkout scm
            }
        }

        stage('Build Image') {
            steps {
                script {
                    docker.build("${IMAGE_NAME}")
                }
            }
        }

        stage('Run Tests') {
            steps {
                sh 'python -m unittest discover tests'
            }
        }

        stage('Push to Docker Hub') {
            steps {
                withDockerRegistry([credentialsId: 'dockerhub-credentials', url: '']) {
                    script {
                        docker.image("${IMAGE_NAME}").push('latest')
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
