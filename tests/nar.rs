use anyhow::Result;
use candle_core::{Device, Tensor};
use yue2::{
    nar::{song_chunks, Chunk},
    protocol::{CODEC_OFFSET, CODEC_SIZE, CONTEXT, MUSIC_END},
};

fn noise(chunks: &[Chunk]) -> Result<Vec<f32>> {
    Ok(
        Tensor::cat(&chunks.iter().map(|c| &c.noise).collect::<Vec<_>>(), 0)?
            .flatten_all()?
            .to_vec1()?,
    )
}

#[test]
fn seeded_noise_precedes_chunk_cuts_and_validates_inputs() -> Result<()> {
    let prefix = [1, 2];
    let codec = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    let one = song_chunks(&prefix, &codec, 831001, 23, &Device::Cpu)?;
    let many = song_chunks(&prefix, &codec, 831001, 13, &Device::Cpu)?;
    assert_eq!(one.len(), 1);
    assert_eq!(
        many.iter()
            .map(|c| c.noise.dim(0).unwrap())
            .collect::<Vec<_>>(),
        [4, 4, 1]
    );
    assert_eq!(noise(&one)?, noise(&many)?);
    assert_eq!(
        many[1].ar_tokens,
        [
            1,
            2,
            CODEC_OFFSET + 5,
            CODEC_OFFSET + 6,
            CODEC_OFFSET + 7,
            CODEC_OFFSET + 8,
            MUSIC_END
        ]
    );
    assert_ne!(
        noise(&one)?,
        noise(&song_chunks(&prefix, &codec, 831002, 23, &Device::Cpu)?)?
    );
    for (p, c, context) in [
        (&[][..], &codec[..], 23),
        (&prefix[..], &[][..], 23),
        (&prefix[..], &[CODEC_SIZE][..], 23),
        (&prefix[..], &codec[..], 6),
        (&prefix[..], &codec[..], CONTEXT + 1),
    ] {
        assert!(song_chunks(p, c, 0, context, &Device::Cpu).is_err());
    }
    Ok(())
}
