import ingest
import split
import transform
import train
import evaluate
import register

if __name__ == "__main__":
    # ingest
    raw_df = ingest.load_raw_as_dataframe()

    # split
    split.create_folds(raw_df)

    # transform
    # clean data set in this case so nothing to do here

    # train
    train.run(1, "rf")

    # evaluate


    # register
    
