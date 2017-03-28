\version "2.10.33"

\header {
  title = "85 - O Gott, du frommer Gott"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key e \major
}


soprano = \relative c'' {
  \partial 4 b4 
  gis4 fis8 e8 b'4 b4 
  cis2 r4 cis4 
  fis,4 fis4 b4 a4 
  gis4 fis4 e4 b'4 
  cis4 cis4 b4 a4   %5
  gis2 r4 fis4 
  gis4 ais4 b4 cis4 
  dis4 cis8 b8 b4 b4 
  b4 b4 e4 d4 
  cis2 r4 cis4   %10
  cis4 cis4 fis4 e4 
  dis2 r4 b4 
  cis4 b4 cis4 dis4 
  e2 r4 b4 
  b4 a8 gis8 fis4 fis4   %15
  e2. 
}

alto = \relative c' {
  \partial 4 e4 
  e4 dis4 e8 fis8 gis4 
  a2 r4 fis8 e8 
  dis8 cis8 dis4 e4 fis4 
  e4 dis4 b4 e4 
  e4 a4 a8 gis8 fis4   %5
  e2 r4 fis4 
  fis4 e4 fis4 gis4 
  fis2 dis4 fis4 
  e4 e4 e8 fis8 gis4 
  a2 r4 a4   %10
  a4 ais4 b4 b8 ais8 
  b2 r4 b4 
  a4 b4 b8 a8 gis8 fis8 
  e2 r4 fis4 
  e4 e4 e4 dis4   %15
  b2. 
}

tenor = \relative c' {
  \partial 4 gis8  a8 
  b4 a4 b4 e4 
  e2 r4 cis4 
  b4 b4 b4 b4 
  b4. a8 gis4 b4 
  a8 cis8 fis8 e8 dis8 e8 e8 dis8   %5
  b2 r4 b4 
  b4 cis4 fis,8 b8 b4 
  b4 ais4 fis4 b8 a8 
  gis8 fis8 gis8 a8 b4 e4 
  e2 r4 e4   %10
  fis4 cis4 dis4 e4 
  fis2 r4 e4 
  e4 e4 a,4 b8 a8 
  gis2 r4 fis8 gis16 a16 
  gis4 a8 b8 cis4 b8 a8   %15
  gis2. 
}

baixo = \relative c {
  \partial 4 e4 
  e4 fis4 gis4 fis8 e8 
  a2 r4 ais4 
  b4 a4 gis4 dis4 
  e4 b4 e4 gis4 
  a4 fis4 b4 b,4   %5
  e2 r4 dis4 
  e4 cis4 dis4 e4 
  fis2 b,4 dis4 
  e8 dis8 e8 fis8 gis4 fis8 e8 
  a2 r4 a8 gis8   %10
  fis4 e4 dis4 cis4 
  b2 r4 gis'4 
  a4 gis4 fis4 b,4 
  cis2 r4 dis4 
  e8 dis8 cis8 b8 a4 b4   %15
  e2. 
}

\score {
  <<
    \new StaffGroup <<
      \override StaffGroup.SystemStartBracket #'style = #'line 
      \new Staff {
        <<
          \global
          \new Voice = "soprano" { \voiceOne \soprano }
          \new Voice = "alto" { \voiceTwo \alto }
        >>
      }
      \new Staff {
        <<
          \global
          \clef "bass"
          \new Voice = "tenor" {\voiceOne \tenor }
          \new Voice = "baixo" { \voiceTwo \baixo \bar "|."}
        >>
      }
    >>
  >>
  \layout {}
  \midi {}
}