from db.connector import DatabaseConnection
import json
import logger  # Import the logger


class DatabaseManager:
    """
    A class to manage the database operations for the application.
    This class provides methods to check if a data source or category exists,
    insert data sources and categories, bulk insert products and reviews,
    and insert results of various analyses into the database.
    It uses the DatabaseConnection class for database interactions.
    """

    def __init__(self, dbname, user, password, host, port):
        self.dbname = dbname
        self.user = user
        self.password = password
        self.host = host
        self.port = port

    def connection(self):
        """
        Establish a connection to the PostgreSQL database.
        """
        return DatabaseConnection(self.dbname, self.user, self.password, self.host, self.port)

    def check_if_data_source_exists(self, data_source_name):
        """
        Check if a data source already exists in the database.
        """
        query = f"""
            SELECT id
            FROM data_sources
            WHERE name = '{data_source_name}';
        """
        try:
            with self.connection() as db:
                data_source_id = db.execute_query(query)
                data_source_id = data_source_id[0][0] if data_source_id else None
                return data_source_id
        except Exception as e:
            logger.error(f"An error occurred while checking if data source exists: {e}")
            raise
        return data_source_id

    def insert_data_source(self, data_source_name, data_source_url):
        """
        Insert a new data source into the database.

        :param data_source_name: Name of the data source.
        :param data_source_url: URL of the data source.
        """
        query = f"""
            INSERT INTO data_sources (name, url)
            VALUES ('{data_source_name}', '{data_source_url}')
            RETURNING id;
        """
        try:
            with self.connection() as db:
                data_source_id = db.execute_query(query)[0][0]
                return data_source_id
        except Exception as e:
            logger.error(f"An error occurred while inserting data source: {e}")
            raise

    def check_if_category_exists(self, category_name, data_source_id):
        """
        Check if a category already exists in the database.

        :param category_name: Name of the category.
        :param data_source_id: ID of the data source.

        Returns the ID of the category if it exists, otherwise None.
        """
        query = f"""
            SELECT id
            FROM categories
            WHERE name = '{category_name}'
            AND data_source_id = {data_source_id};
        """
        try:
            with self.connection() as db:
                category_id = db.execute_query(query)
                category_id = category_id[0][0] if category_id else None
                return category_id
        except Exception as e:
            logger.error(f"An error occurred while checking if category exists: {e}")
            raise

    def insert_category(self, category_name, data_source_id):
        """
        Insert a new category into the database.

        :param category_name: Name of the category.
        :param data_source_id: ID of the data source.
        """

        try:
            with self.connection() as db:
                category_id = db.insert_rows('categories',
                                             [{'name': category_name, 'data_source_id': data_source_id}],
                                             ['name', 'data_source_id'])[0][0]
                return category_id
        except Exception as e:
            logger.error(f"An error occurred while inserting category: {e}")
            raise

    def bulk_insert_products(self, product_data, category_id):
        """
        Bulk insert products into the database.

        :param product_data: DataFrame containing product data. Must have columns 'id', 'name', 'description'.
        :param category_id: ID of the category to which the products belong.

        Returns a dictionary mapping external product IDs to internal product IDs.
        """
        product_data = product_data[['id', 'name', 'description']].rename(
            columns={'id': 'external_id'})
        product_data['name'] = product_data['name'].astype(str).apply(lambda x: x[:200])
        product_data['category_id'] = category_id

        columns = ['category_id', 'name', 'description', 'external_id']
        product_data = product_data[columns]
        rows = product_data.to_dict('records')
        return_columns = ['id', 'external_id']

        try:
            with self.connection() as db:
                product_id_mapping = db.insert_rows('products', rows, columns, return_columns)
        except Exception as e:
            logger.error(f"An error occurred while inserting products: {e}")
            raise
        product_id_mapping = {row[1]: row[0] for row in product_id_mapping}
        return product_id_mapping

    def bulk_insert_reviews(self, review_data, product_id_mapping):
        """
        Bulk insert reviews into the database.

        :param review_data: DataFrame containing review data. Must have columns 'product_id', 'content', and 'rating'.
        :param product_id_mapping: Dictionary mapping external product IDs to internal product IDs.

        """
        review_data['product_id'] = review_data['product_id'].astype(str).map(product_id_mapping)
        review_data = review_data[['product_id', 'content', 'rating']]
        columns = review_data.columns.tolist()
        rows = review_data.to_dict('records')

        try:
            with self.connection() as db:
                db.insert_rows('reviews', rows, columns)
        except Exception as e:
            logger.error(f"An error occurred while inserting reviews: {e}")
            raise

    def insert_aspect_extraction(self,
                                 category_id,
                                 aspect_counter,
                                 name,
                                 method,
                                 description,
                                 status='completed',
                                 message='Aspect extraction completed successfully.',
                                 retry_count=0,
                                 top_k_aspects=100):
        """
        Insert aspect extraction results into the database.
        """

        try:
            with self.connection() as db:
                aspect_extraction_id = db.insert_rows(
                    'aspect_extractions',
                    rows=[{
                        'category_id': category_id,
                        'status': status,
                        'message': message,
                        'retry_count': retry_count,
                        'name': name,
                        'method': method,
                        'description': description
                    }],
                    columns=['category_id',
                             'status',
                             'message',
                             'retry_count',
                             'name',
                             'method',
                             'description'])[0][0]
                aspect_rows = [{'aspect_extraction_id': aspect_extraction_id,
                                'aspect': aspect,
                                'frequency': count}
                               for aspect, count
                               in aspect_counter.most_common(top_k_aspects)]
                aspect_cols = ['aspect_extraction_id', 'aspect', 'frequency']
                db.insert_rows('extracted_aspects', aspect_rows, aspect_cols)
        except Exception as e:
            logger.error(f"An error occurred while inserting aspect extraction results: {e}")
            raise

    def insert_ontology_extraction(self,
                                   aspect_extraction_id,
                                   ontology_tree,
                                   synsets,
                                   name,
                                   method,
                                   description,
                                   status='completed',
                                   message='Ontology extraction completed successfully.',
                                   retry_count=0):
        """
        Insert ontology extraction results into the database.
        """
        try:
            with self.connection() as db:
                ontology_extraction_id = db.insert_rows(
                    'ontology_extractions',
                    rows=[{
                        'aspect_extraction_id': aspect_extraction_id,
                        'status': status,
                        'message': message,
                        'retry_count': retry_count,
                        'name': name,
                        'method': method,
                        'description': description
                    }],
                    columns=['aspect_extraction_id',
                             'status',
                             'message',
                             'retry_count',
                             'name',
                             'method',
                             'description'],)[0][0]
                ontology_tree = json.dumps(ontology_tree)
                synsets = json.dumps(synsets)
                ontology_tree_rows = [{'ontology_extraction_id': ontology_extraction_id,
                                       'ontology': ontology_tree,
                                       'synsets': synsets}]
                ontology_tree_cols = ['ontology_extraction_id', 'ontology', 'synsets']
                db.insert_rows('ontology_trees', ontology_tree_rows, ontology_tree_cols)
                return ontology_extraction_id
        except Exception as e:
            logger.error(f"An error occurred while inserting ontology extraction results: {e}")
            raise

    def insert_argumentative_analysis(self,
                                      ontology_extraction_id,
                                      argumentative_analysis_df,
                                      name,
                                      method,
                                      description,
                                      status='completed',
                                      message='Argumentative analysis completed successfully.',
                                      retry_count=0):
        """
        Insert argumentative analysis results into the database.

        :param ontology_extraction_id: ID of the ontology extraction.
        :param argumentative_analysis_df: DataFrame containing argumentative analysis results. Must have columns 'result_tree', 'product_id'.
        """
        try:
            with self.connection() as db:
                argumentative_analysis_id = db.insert_rows(
                    'argumentative_analyses',
                    rows=[{
                        'ontology_extraction_id': ontology_extraction_id,
                        'status': status,
                        'message': message,
                        'retry_count': retry_count,
                        'name': name,
                        'method': method,
                        'description': description
                    }],
                    columns=['ontology_extraction_id',
                             'status',
                             'message',
                             'retry_count',
                             'name',
                             'method',
                             'description'],)[0][0]
                result_cols = ['argumentative_analysis_id',
                               'product_id',
                               'aspect',
                               'polarity',
                               'strength',
                               'strongest_support_phrase',
                               'strongest_attack_phrase',
                               'strongest_support_feature',
                               'strongest_attack_feature']
                argumentative_analysis_df['argumentative_analysis_id'] = argumentative_analysis_id
                result_rows = argumentative_analysis_df[result_cols].to_dict('records')
                db.insert_rows('arguments', result_rows, result_cols)
        except Exception as e:
            logger.error(f"An error occurred while inserting argumentative analysis results: {e}")
            raise

    def query_all_reviews(self, data_source_name, category_name):
        """
        Query all reviews for a given data source and given category.
        """
        # first let's get category id with largest id so the latest
        # get the max category id
        query = f"""
            SELECT id
            FROM categories
            WHERE data_source_id = (SELECT id FROM data_sources WHERE name = '{data_source_name}')
            AND name = '{category_name}'
            ORDER BY id DESC
            LIMIT 1;
        """
        try:
            with self.connection() as db:
                category_id = db.execute_query(query)
                category_id = category_id[0][0] if category_id else None
        except Exception as e:
            logger.error(f"An error occurred while querying category id for getting reviews: {e}")
            raise

        # now get all reviews for this category
        query = f"""
            SELECT reviews.id, reviews.product_id, reviews.content, reviews.rating
            FROM reviews
            JOIN products ON reviews.product_id = products.id
            WHERE products.category_id = {category_id};
        """
        try:
            with self.connection() as db:
                reviews = db.execute_query(query)
        except Exception as e:
            logger.error(f"An error occurred while querying reviews: {e}")
            raise
        return category_id, reviews

    def query_aspect_extraction_by_name(self, category_id, aspect_extraction_name):
        """
        Query aspect extraction by name.
        """
        query = f"""
            SELECT id
            FROM aspect_extractions
            WHERE category_id = {category_id}
            AND name = '{aspect_extraction_name}'
            ORDER BY id DESC
            LIMIT 1;
        """
        try:
            with self.connection() as db:
                aspect_extraction_id = db.execute_query(query)
                aspect_extraction_id = aspect_extraction_id[0][0] if aspect_extraction_id else None
        except Exception as e:
            logger.error(f"An error occurred while querying aspect extraction id: {e}")
            raise
        return aspect_extraction_id

    def query_ontology_extraction_by_name(self, aspect_extraction_id, ontology_extraction_name):
        # now get ontology extraction id
        query = f"""
            SELECT id
            FROM ontology_extractions
            WHERE aspect_extraction_id = {aspect_extraction_id}
            AND name = '{ontology_extraction_name}'
            ORDER BY id DESC
            LIMIT 1;
        """
        try:
            with self.connection() as db:
                ontology_extraction_id = db.execute_query(query)
                ontology_extraction_id = ontology_extraction_id[0][0] if ontology_extraction_id else None
        except Exception as e:
            logger.error(f"An error occurred while querying ontology extraction id: {e}")
            raise
        return ontology_extraction_id

    def query_aspects_by_aspect_extraction(self, category_id, aspect_extraction_name):
        """
        Query aspects by aspect extraction name.
        """
        # first get aspect extraction id
        aspect_extraction_id = self.query_aspect_extraction_by_name(category_id, aspect_extraction_name)
        if not aspect_extraction_id:
            raise ValueError(f"No aspect extraction found for name {aspect_extraction_name}")

        # now get aspects
        query = f"""
            SELECT aspect, frequency
            FROM extracted_aspects
            WHERE aspect_extraction_id = {aspect_extraction_id};
        """
        try:
            with self.connection() as db:
                aspects = db.execute_query(query)
        except Exception as e:
            logger.error(f"An error occurred while querying aspects: {e}")
            raise
        aspect_frequency = {aspect: frequency for aspect, frequency in aspects}
        return aspect_extraction_id, aspect_frequency

    def query_ontology_by_ontology_extraction(self, category_id, aspect_extraction_name, ontology_extraction_name):
        """
        Query ontology by ontology extraction name.
        """
        # first get aspect extraction id
        aspect_extraction_id = self.query_aspect_extraction_by_name(category_id, aspect_extraction_name)
        if not aspect_extraction_id:
            raise ValueError(f"No aspect extraction found for name {aspect_extraction_name}")

        # now get ontology extraction id
        ontology_extraction_id = self.query_ontology_extraction_by_name(aspect_extraction_id, ontology_extraction_name)
        if not ontology_extraction_id:
            raise ValueError(f"No ontology extraction found for name {ontology_extraction_name}")

        # now get ontology tree
        query = f"""
            SELECT ontology, synsets
            FROM ontology_trees
            WHERE ontology_extraction_id = {ontology_extraction_id};
        """
        try:
            with self.connection() as db:
                res = db.execute_query(query)
                ontology_tree, synsets = res[0]
        except Exception as e:
            logger.error(f"An error occurred while querying ontology tree: {e}")
            raise
        return ontology_extraction_id, ontology_tree, synsets

    def update_aspect_llm_judge_score(self, aspect_extraction_id: int, aspect: str, score: float):
        """
        Update the LLM judge score for a specific aspect.

        Args:
            aspect_extraction_id: ID of the aspect extraction
            aspect: The aspect term to update
            score: The LLM judge score to set
        """
        query = f"""
            UPDATE extracted_aspects 
            SET llm_judge_score = {score},
                score_updated_at = CURRENT_TIMESTAMP
            WHERE aspect_extraction_id = {aspect_extraction_id}
            AND aspect = '{aspect}'
            RETURNING id;
        """
        try:
            with self.connection() as db:
                db.execute_query(query)
        except Exception as e:
            logger.error(f"An error occurred while updating LLM judge score: {e}")
            raise

    def update_ontology_llm_judge_score(self, ontology_extraction_id: int, score: float):
        """
        Update the LLM judge score for a specific ontology extraction.

        Args:
            ontology_extraction_id: ID of the ontology extraction
            score: The LLM judge score to set
        """
        query = f"""
            UPDATE ontology_trees 
            SET llm_judge_score = {score},
                score_updated_at = CURRENT_TIMESTAMP
            WHERE ontology_extraction_id = {ontology_extraction_id}
            RETURNING id;
        """
        try:
            with self.connection() as db:
                db.execute_query(query)
        except Exception as e:
            logger.error(f"An error occurred while updating LLM judge score for ontology: {e}")
            raise
