\version "2.12.0"
\header {
  title = "Easter Oratorio, II"
  composer = "Bach"
}

\new PianoStaff <<
  \new Staff {
    \relative c' {
      \time 3/4
      \key d \major
      <fis b d>8.
      <g b cis>16
      <fis b d>4
      <fis ais cis>4
      <d b'>4
      r8
      <e b' e>8
      <e b' e>4
      <fis a dis>4
      r8
      <fis a dis>8
      <fis a dis>4
      <g b e>4
      r8
      <g c! g'>8
      <g c g'>4
      <a c! fis>
    }
  }
  \new Staff {
    \relative c {
      \clef bass
      \key d \major
      \time 3/4
      d8. e16 fis8.[ fis,16 fis8. fis16]
      g4 r8 g'8 g8. g16
      fis4 r8 fis8 fis8. fis16
      e4 r8 e8 e8. e16
      d4
    }
  }
>>
