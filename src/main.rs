use std::fs;

use kd_tree::{KdPoint};
use rust_bert::pipelines::sentence_embeddings::{
    SentenceEmbeddingsBuilder, SentenceEmbeddingsModelType,
};
use serde::{Deserialize};


#[derive(Debug, Deserialize)]
pub struct Library {
    pub books: Vec<Book>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct Book {
    pub title: String,

    pub author: String,

    pub summary: String,
}

impl Book {
    fn to_embedded(self, embeddings: [f32; 384]) -> EmbeddedBook {
        EmbeddedBook {
            title: self.title,
            author: self.author,
            summary: self.summary,
            embeddings: embeddings,
        }
    }
}
#[derive(Debug)]
pub struct EmbeddedBook {
    pub title: String,

    pub author: String,

    pub summary: String,

    pub embeddings: [f32; 384],
}

impl KdPoint for EmbeddedBook {
    type Scalar = f32;
    type Dim = typenum::U2; // 2 dimensional tree.
    fn at(&self, k: usize) -> f32 {
        self.embeddings[k]
    }
}

fn main() -> anyhow::Result<()> {
    // Set-up sentence embeddings model
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;

    let config = fs::read_to_string("data/books.json")?;
    let library: Library = serde_json::from_str(&config)?;
    let mut embeddedbooks = Vec::new();
    for book in library.books.clone() {
        let embeddings = model.encode(&[book.clone().summary])?;

        embeddedbooks.push(book.to_embedded(pop(embeddings[0].as_slice())));
    }

    let first_book = &library.books[0];
    let embeddings = model.encode(&["young millionaire"])?;
    let point = first_book
        .clone()
        .to_embedded(pop(embeddings[0].as_slice()));
        
    let kdtree = kd_tree::KdSlice::sort_by(&mut embeddedbooks, |item1, item2, k| {
        item1.embeddings[k]
            .partial_cmp(&item2.embeddings[k])
            .unwrap()
    });

    let nearests = kdtree.nearests(&point, 10);
    for nearest in nearests {
        println!("nearest: {:?}", nearest.item.title);
        println!("distance: {:?}", nearest.squared_distance);
    }

    Ok(())
}

fn pop(barry: &[f32]) -> [f32; 384] {
    barry.try_into().expect("slice with incorrect length")
}
