pipeline {
    agent any

    environment {
        IMAGE_NAME = 'milk-quality'
    }

    stages {
        stage('Clone Repository') {
             steps {
                git branch: 'main', url: 'https://github.com/Atul-728/Milk_Quality_Prediction.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t %IMAGE_NAME% .'
            }
        }

        stage('Run Script in Container') {
            steps {
                bat 'docker run --rm %IMAGE_NAME%'
            }
        }
    }

    post {
        always {
            echo 'Pipeline completed!'
        }
    }
}
