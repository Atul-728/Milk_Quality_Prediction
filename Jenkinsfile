pipeline {
    agent any

    environment {
        IMAGE_NAME = 'milk-quality'
        CONTAINER_NAME = 'milk_quality_container'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git branch: 'main', url: 'https://github.com/Atul-728/Milk_Quality_Prediction.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker build -t %%IMAGE_NAME%% .'
            }
        }

        stage('Run Container') {
            steps {
                bat '''
                    docker rm -f %%CONTAINER_NAME%% 2>nul || echo No container to remove
                    docker run -d --name %%CONTAINER_NAME%% %%IMAGE_NAME%%
                '''
            }
        }

        stage('Check Logs') {
            steps {
                bat 'docker logs %%CONTAINER_NAME%%'
            }
        }
    }

    post {
        always {
            echo 'Pipeline completed!'
        }
    }
}
