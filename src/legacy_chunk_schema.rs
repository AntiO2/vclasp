#![allow(dead_code)]

pub const FILE_IDENTIFIER: &str = "HVS1";

#[derive(Copy, Clone, PartialEq)]
pub struct ChunkHeader<'a> {
    pub _tab: flatbuffers::Table<'a>,
}

impl<'a> flatbuffers::Follow<'a> for ChunkHeader<'a> {
    type Inner = ChunkHeader<'a>;

    #[inline]
    unsafe fn follow(buf: &'a [u8], loc: usize) -> Self::Inner {
        Self {
            _tab: flatbuffers::Table::new(buf, loc),
        }
    }
}

impl<'a> ChunkHeader<'a> {
    pub const VT_MAGIC: flatbuffers::VOffsetT = 4;
    pub const VT_FORMAT_VERSION: flatbuffers::VOffsetT = 6;
    pub const VT_CODEC: flatbuffers::VOffsetT = 8;
    pub const VT_WIDTH: flatbuffers::VOffsetT = 10;
    pub const VT_HEIGHT: flatbuffers::VOffsetT = 12;
    pub const VT_FPS_NUM: flatbuffers::VOffsetT = 14;
    pub const VT_FPS_DEN: flatbuffers::VOffsetT = 16;
    pub const VT_SPS_PPS_LENGTH: flatbuffers::VOffsetT = 18;
    pub const VT_PAYLOAD_LENGTH: flatbuffers::VOffsetT = 20;
    pub const VT_INDEX_LENGTH: flatbuffers::VOffsetT = 22;
    pub const VT_CREATED_AT: flatbuffers::VOffsetT = 24;

    #[inline]
    pub fn magic(&self) -> Option<&'a str> {
        unsafe {
            self._tab
                .get::<flatbuffers::ForwardsUOffset<&str>>(Self::VT_MAGIC, None)
        }
    }

    #[inline]
    pub fn format_version(&self) -> u16 {
        unsafe {
            self._tab
                .get::<u16>(Self::VT_FORMAT_VERSION, Some(0))
                .unwrap()
        }
    }

    #[inline]
    pub fn codec(&self) -> Option<&'a str> {
        unsafe {
            self._tab
                .get::<flatbuffers::ForwardsUOffset<&str>>(Self::VT_CODEC, None)
        }
    }

    #[inline]
    pub fn width(&self) -> u16 {
        unsafe { self._tab.get::<u16>(Self::VT_WIDTH, Some(0)).unwrap() }
    }

    #[inline]
    pub fn height(&self) -> u16 {
        unsafe { self._tab.get::<u16>(Self::VT_HEIGHT, Some(0)).unwrap() }
    }

    #[inline]
    pub fn fps_num(&self) -> u16 {
        unsafe { self._tab.get::<u16>(Self::VT_FPS_NUM, Some(0)).unwrap() }
    }

    #[inline]
    pub fn fps_den(&self) -> u16 {
        unsafe { self._tab.get::<u16>(Self::VT_FPS_DEN, Some(0)).unwrap() }
    }

    #[inline]
    pub fn sps_pps_length(&self) -> u64 {
        unsafe {
            self._tab
                .get::<u64>(Self::VT_SPS_PPS_LENGTH, Some(0))
                .unwrap()
        }
    }

    #[inline]
    pub fn payload_length(&self) -> u64 {
        unsafe {
            self._tab
                .get::<u64>(Self::VT_PAYLOAD_LENGTH, Some(0))
                .unwrap()
        }
    }

    #[inline]
    pub fn index_length(&self) -> u64 {
        unsafe {
            self._tab
                .get::<u64>(Self::VT_INDEX_LENGTH, Some(0))
                .unwrap()
        }
    }

    #[inline]
    pub fn created_at(&self) -> u64 {
        unsafe { self._tab.get::<u64>(Self::VT_CREATED_AT, Some(0)).unwrap() }
    }
}

pub struct ChunkHeaderArgs<'a> {
    pub magic: flatbuffers::WIPOffset<&'a str>,
    pub format_version: u16,
    pub codec: flatbuffers::WIPOffset<&'a str>,
    pub width: u16,
    pub height: u16,
    pub fps_num: u16,
    pub fps_den: u16,
    pub sps_pps_length: u64,
    pub payload_length: u64,
    pub index_length: u64,
    pub created_at: u64,
}

pub struct ChunkHeaderBuilder<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> {
    fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
    start: flatbuffers::WIPOffset<flatbuffers::TableUnfinishedWIPOffset>,
}

impl<'a: 'b, 'b, A: flatbuffers::Allocator + 'a> ChunkHeaderBuilder<'a, 'b, A> {
    #[inline]
    pub fn new(fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>) -> Self {
        let start = fbb.start_table();
        Self { fbb, start }
    }

    #[inline]
    pub fn add_magic(&mut self, value: flatbuffers::WIPOffset<&'b str>) {
        self.fbb
            .push_slot_always::<flatbuffers::WIPOffset<_>>(ChunkHeader::VT_MAGIC, value);
    }

    #[inline]
    pub fn add_format_version(&mut self, value: u16) {
        self.fbb
            .push_slot::<u16>(ChunkHeader::VT_FORMAT_VERSION, value, 0);
    }

    #[inline]
    pub fn add_codec(&mut self, value: flatbuffers::WIPOffset<&'b str>) {
        self.fbb
            .push_slot_always::<flatbuffers::WIPOffset<_>>(ChunkHeader::VT_CODEC, value);
    }

    #[inline]
    pub fn add_width(&mut self, value: u16) {
        self.fbb.push_slot::<u16>(ChunkHeader::VT_WIDTH, value, 0);
    }

    #[inline]
    pub fn add_height(&mut self, value: u16) {
        self.fbb.push_slot::<u16>(ChunkHeader::VT_HEIGHT, value, 0);
    }

    #[inline]
    pub fn add_fps_num(&mut self, value: u16) {
        self.fbb.push_slot::<u16>(ChunkHeader::VT_FPS_NUM, value, 0);
    }

    #[inline]
    pub fn add_fps_den(&mut self, value: u16) {
        self.fbb.push_slot::<u16>(ChunkHeader::VT_FPS_DEN, value, 0);
    }

    #[inline]
    pub fn add_sps_pps_length(&mut self, value: u64) {
        self.fbb
            .push_slot::<u64>(ChunkHeader::VT_SPS_PPS_LENGTH, value, 0);
    }

    #[inline]
    pub fn add_payload_length(&mut self, value: u64) {
        self.fbb
            .push_slot::<u64>(ChunkHeader::VT_PAYLOAD_LENGTH, value, 0);
    }

    #[inline]
    pub fn add_index_length(&mut self, value: u64) {
        self.fbb
            .push_slot::<u64>(ChunkHeader::VT_INDEX_LENGTH, value, 0);
    }

    #[inline]
    pub fn add_created_at(&mut self, value: u64) {
        self.fbb
            .push_slot::<u64>(ChunkHeader::VT_CREATED_AT, value, 0);
    }

    #[inline]
    pub fn finish(self) -> flatbuffers::WIPOffset<ChunkHeader<'a>> {
        let offset = self.fbb.end_table(self.start);
        flatbuffers::WIPOffset::new(offset.value())
    }
}

pub fn create_chunk_header<'a: 'b, 'b, A: flatbuffers::Allocator + 'a>(
    fbb: &'b mut flatbuffers::FlatBufferBuilder<'a, A>,
    args: &ChunkHeaderArgs<'b>,
) -> flatbuffers::WIPOffset<ChunkHeader<'a>> {
    let mut builder = ChunkHeaderBuilder::new(fbb);
    builder.add_magic(args.magic);
    builder.add_format_version(args.format_version);
    builder.add_codec(args.codec);
    builder.add_width(args.width);
    builder.add_height(args.height);
    builder.add_fps_num(args.fps_num);
    builder.add_fps_den(args.fps_den);
    builder.add_sps_pps_length(args.sps_pps_length);
    builder.add_payload_length(args.payload_length);
    builder.add_index_length(args.index_length);
    builder.add_created_at(args.created_at);
    builder.finish()
}

pub fn root_as_chunk_header(buf: &[u8]) -> ChunkHeader<'_> {
    unsafe { flatbuffers::root_unchecked::<ChunkHeader<'_>>(buf) }
}
