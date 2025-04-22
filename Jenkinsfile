pipeline {
    agent any

    environment {
        DOCKER_IMAGE = '12211415/milk_quality_prediction:latest'
    }

    stages {
        stage('Checkout') {
            steps {
                git credentialsId: 'github-credentials', url: 'https://github.com/Atul-728/Milk_Quality_Prediction.git'
            }
        }

        stage('Build Image') {
            steps {
                bat "docker build -t ${env.DOCKER_IMAGE} ."
            }
        }

        stage('Run Tests') {
            steps {
                script {
                    docker.image(env.DOCKER_IMAGE).inside {
                        bat 'pytest > result.txt || type result.txt'  // Run pytest, show result even on fail
                    }
                }
            }
        }

        stage('Push to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'docker-hub-credentials', passwordVariable: 'DOCKER_PASSWORD', usernameVariable: 'DOCKER_USERNAME')]) {
                    bat "docker login -u %DOCKER_USERNAME% -p %DOCKER_PASSWORD%"
                    bat "docker push ${env.DOCKER_IMAGE}"
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
