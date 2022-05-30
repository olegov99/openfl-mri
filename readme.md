# Federated learning system for learning MRI Images classifer

This project was created for learning MRI images classifier using the Federated Learning method ([McMahan B., et al., 2016](https://arxiv.org/abs/1602.05629)). 

The project structure consists of two following folders:
- **distributed_federated_system** - contains the implemented federated system, which can be deployed and run on multiple nodes.
- **experimtents** - includes federated learning experiments using differently distributed data between clients

## How to run the federated system?

In order to run the federated system on multiple different nodes, please, take the following steps:
- Move every single envoy_#  and director (aggregator) + workspace folders to separate nodes and install required packages using ```pip install -r requirements.txt```
- For every envoy_# download its corresponding data from [here](https://drive.google.com/drive/folders/1AK5SGLVvkVwENcU2LSI1VudoYdnfWqfJ?usp=sharing) and extract all files to the according envoy's **data** folder.
- On aggregator node:
    - Set ```listen_host``` and ```listen_port``` parameters in ***director.yaml***
    - Open **director** folder in cmd or bash and run the following command to start the director:

        ```fx director start --disable-tls -c director_config.yaml```
- On envoys nodes:
    - Open **envoy_#** folder in cmd or bash and run the following command to start the envoy:
        
        ```fx envoy start -n ${ENVOY_NAME} --disable-tls --envoy-config-path envoy_config.yaml -dh ${AGGREGATOR_FQDN} -dp ${AGGREGATOR_PORT}```
- Open ***experiment-manager.ipynb*** in Jupyter Notebook and run all of its cells.


## Some of the experiments data

Loss function for experiment with 4 collaborators using distributed system and model from [Qu R., et al.](https://www.mdpi.com/2078-2489/13/3/124)

![Loss Function](https://github.com/olegov99/openfl-mri/blob/master/experiments/loss_4_col_openfl.png)

Loss function for experiment with 4 collaborators using simple CNN.

![Loss Function](https://github.com/olegov99/openfl-mri/blob/master/experiments/4_col_loss.png)

Loss function for experiment with 3 collaborators using simple CNN.

![Loss Function](https://github.com/olegov99/openfl-mri/blob/master/experiments/3_col_loss.png)

Client_1 train dataset distribution:

![Data Distribution](https://github.com/olegov99/openfl-mri/blob/master/experiments/clients_data_distribution/client_1_train.png)

Client_1 test dataset distribution:

![Data Distribution](https://github.com/olegov99/openfl-mri/blob/master/experiments/clients_data_distribution/client_1_test.png)

Client_2 train dataset distribution:

![Data Distribution](https://github.com/olegov99/openfl-mri/blob/master/experiments/clients_data_distribution/client_2_train.png)

Client_2 test dataset distribution:

![Data Distribution](https://github.com/olegov99/openfl-mri/blob/master/experiments/clients_data_distribution/client_2_test.png)

Client_3 train dataset distribution:

![Data Distribution](https://github.com/olegov99/openfl-mri/blob/master/experiments/clients_data_distribution/client_3_train.png)

Client_3 test dataset distribution:

![Data Distribution](https://github.com/olegov99/openfl-mri/blob/master/experiments/clients_data_distribution/client_3_test.png)

Client_4 train dataset distribution:

![Data Distribution](https://github.com/olegov99/openfl-mri/blob/master/experiments/clients_data_distribution/client_4_train.png)

Client_4 test dataset distribution:

![Data Distribution](https://github.com/olegov99/openfl-mri/blob/master/experiments/clients_data_distribution/client_4_test.png)




