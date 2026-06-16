//! On-close compaction: rewrite each `<table>.arrow` IPC stream into a
//! `<table>.parquet` file (zstd, dictionary-encoded), then bundle the directory
//! into a single STORED zip (`<dir>.dieslog`) for sharing.
//!
//! Parquet is the analysis-grade artifact (`pd.read_parquet`); the zip is
//! transport-only (extract to use). The intermediate `.arrow` files are removed
//! once their Parquet counterpart is written.

use std::fs::{self, File};
use std::io::BufReader;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use arrow::ipc::reader::StreamReader;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

use crate::schema;

/// Compact every table's IPC stream to Parquet (removing the `.arrow` source),
/// then bundle the directory into a STORED zip. Returns the zip path.
pub fn finalize(dir: &Path) -> Result<PathBuf> {
    for &table in schema::TABLES {
        let arrow_path = dir.join(format!("{table}.arrow"));
        if !arrow_path.exists() {
            continue;
        }
        let parquet_path = dir.join(format!("{table}.parquet"));
        if let Err(e) = arrow_to_parquet(&arrow_path, &parquet_path) {
            // Keep the .arrow file as a fallback if conversion fails.
            return Err(e).with_context(|| format!("compacting {table}"));
        }
        let _ = fs::remove_file(&arrow_path);
    }
    zip_dir(dir)
}

fn arrow_to_parquet(arrow_path: &Path, parquet_path: &Path) -> Result<()> {
    let reader = StreamReader::try_new(BufReader::new(File::open(arrow_path)?), None)?;
    let schema = reader.schema();

    // Cap row groups at 64k rows so the lazy reader's window (one row group) is a
    // small, seek-friendly unit (~13s of dense debug data) rather than the ~1M-row
    // default. Frames may straddle a row-group boundary; the reader handles that by
    // gathering a frame's rows across the (≤2) covering row groups.
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(ZstdLevel::try_new(3)?))
        .set_max_row_group_row_count(Some(65536))
        .build();
    let mut writer = ArrowWriter::try_new(File::create(parquet_path)?, schema, Some(props))?;
    for batch in reader {
        writer.write(&batch?)?;
    }
    writer.close()?;
    Ok(())
}

/// Bundle every file in `dir` into a STORED (uncompressed) zip next to it. The
/// zip is transport-only: Parquet already compresses internally, so STORED keeps
/// per-member bytes verbatim and avoids redundant work.
fn zip_dir(dir: &Path) -> Result<PathBuf> {
    use zip::write::SimpleFileOptions;
    use zip::CompressionMethod;

    let zip_path = PathBuf::from(format!("{}.dieslog", dir.display()));
    let mut zw = zip::ZipWriter::new(File::create(&zip_path)?);
    let options = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);

    for entry in fs::read_dir(dir)? {
        let path = entry?.path();
        if !path.is_file() {
            continue;
        }
        let name = path
            .file_name()
            .and_then(|n| n.to_str())
            .context("non-utf8 file name in log dir")?
            .to_string();
        zw.start_file(name, options)?;
        let mut f = File::open(&path)?;
        std::io::copy(&mut f, &mut zw)?;
    }
    zw.finish()?;
    Ok(zip_path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::FrameRecord;
    use crate::meta::MetaJson;
    use crate::writer::LogWriter;
    use dies_core::{mock_world_data, DebugMap};
    use std::time::Instant;

    #[test]
    fn compaction_produces_readable_parquet_and_zip() {
        let tmp = tempfile::tempdir().unwrap();
        let dir = tmp.path().join("session");
        let meta = MetaJson::new(0.0, true, None, None, "yellow_on_positive".into());
        let mut w = LogWriter::open(dir.clone(), meta, Instant::now()).unwrap();
        let world = mock_world_data();
        let debug = DebugMap::new();
        for fid in 0..5 {
            w.push_frame(&FrameRecord::from_world(fid, &world, &debug));
        }
        w.close(Instant::now()).unwrap();

        let zip = finalize(&dir).unwrap();
        assert!(zip.exists(), "zip not created");
        // .arrow removed, .parquet present
        assert!(!dir.join("frames.arrow").exists());
        assert!(dir.join("frames.parquet").exists());

        // Parquet is readable and has the rows we wrote.
        let file = File::open(dir.join("frames.parquet")).unwrap();
        let builder =
            parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
        let reader = builder.build().unwrap();
        let rows: usize = reader.map(|b| b.unwrap().num_rows()).sum();
        assert_eq!(rows, 5);
    }
}
