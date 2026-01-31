from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # Step 1: Ingest data
    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    # Step 2: Transform data
    transformer = DataTransformation()
    train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(train_data, test_data)

    # Step 3: Train model
    trainer = ModelTrainer()
    trainer.initiate_model_trainer(train_arr, test_arr)
