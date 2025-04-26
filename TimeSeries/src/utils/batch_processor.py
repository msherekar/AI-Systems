import pandas as pd
import numpy as np
import os
import logging

class BatchProcessor:
    """
    Process large datasets in batches.
    """

    def __init__(self, file_path, batch_size=10000, logger=None):
        self.file_path = file_path
        self.batch_size = batch_size
        self.logger = logger or logging.getLogger(__name__)

    def _get_batches(self):
        """Private method to yield batches from the CSV file."""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")

        chunksize = self.batch_size
        for chunk in pd.read_csv(self.file_path, chunksize=chunksize):
            yield chunk

    def process_in_batches(self, process_func, output_path=None):
        results = []

        try:
            for i, batch in enumerate(self._get_batches(), 1):
                self.logger.info(f"Processing batch {i}")
                result = process_func(batch)

                if not isinstance(result, (pd.DataFrame, pd.Series)):
                    if isinstance(result, (int, float, np.number)):
                        result = pd.Series([result])
                    else:
                        result = pd.DataFrame(result)

                results.append(result)

            if not results:
                return pd.DataFrame()

            combined_results = pd.concat(results, ignore_index=True)

            if output_path:
                combined_results.to_csv(output_path, index=False)

            return combined_results
        except Exception as e:
            self.logger.error(f"Error in batch processing: {str(e)}")
            raise

    def process_time_series_in_batches(self, date_column, values_column, aggregation='mean', group_by=None):
        all_data = pd.DataFrame()

        for i, batch in enumerate(self._get_batches(), 1):
            self.logger.info(f"Processing batch {i}")
            batch[date_column] = pd.to_datetime(batch[date_column], errors='coerce')
            all_data = pd.concat([all_data, batch], ignore_index=True)

        all_data = all_data.dropna(subset=[date_column])

        if group_by and group_by in all_data.columns:
            grouped = all_data.groupby([date_column, group_by])
        else:
            grouped = all_data.groupby(date_column)

        if aggregation == 'sum':
            result = grouped[values_column].sum().reset_index()
        elif aggregation == 'mean':
            result = grouped[values_column].mean().reset_index()
        else:
            result = grouped[values_column].agg(aggregation).reset_index()

        return result
