use vclasp::{plan_ranges, RecordRange};

fn main() -> Result<(), String> {
    let records = vec![
        RecordRange {
            record_id: 1,
            offset: 0,
            length: 1024,
        },
        RecordRange {
            record_id: 2,
            offset: 2048,
            length: 1024,
        },
    ];

    let plans = plan_ranges(&records, Some(16 * 1024), None)?;
    assert_eq!(plans.len(), 1);
    println!(
        "planned {} range covering {} records",
        plans.len(),
        records.len()
    );
    Ok(())
}
