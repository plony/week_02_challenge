-- Table schema for storing app reviews
CREATE TABLE fintech_reviews (
    review_id NUMBER PRIMARY KEY,
    app_name VARCHAR2(100),
    rating NUMBER,
    sentiment VARCHAR2(10),
    theme VARCHAR2(100),
    review_text CLOB,
    date_posted DATE,
    source VARCHAR2(50)
);