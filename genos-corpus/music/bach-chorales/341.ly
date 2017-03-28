\version "2.10.33"

\header {
  title = "341 - Ich dank dir, lieber Herre"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 a4 
    a4 a4 a4 b4 
  g4 fis4 e4 b'4 
  cis4 b4 a4 gis8 fis8 
  gis4 fis4 e4 e4 
  a4 a4 a4 b4   %5
  gis4 fis4 e4 b'4 
  cis4 b4 a4 gis8 fis8 
  gis4 fis4 e4 e'4 
  d4 cis4 b4 a4 
  a8 b8 cis4 b4 cis4   %10
  d4 cis4 b4 ais4 
  b2. e,4 
  a4 b4 cis4 d4 
  e4 d8 cis8 b4 d4 
  cis4 b4 e4. d8   %15
  cis8 b8 a8 b8 cis4 b4 
  a2. 
}

alto = \relative c' {
  \partial 4 cis8  d8 
    e4 e4 fis4 fis4 
  e4 dis4 b4 gis'4 
  a4 gis4 cis,8 dis8 e4~ 
  e4 dis4 b4 b4 
  cis8 d8 e4 dis8 e8 fis4   %5
  b,8 e4 dis8 e4 e4 
  e4 e4 e4. dis8 
  cis4 dis4 b4 gis'4 
  fis4 e8 fis8 gis4 a8 g8 
  fis8 gis8 a4 gis4 ais4   %10
  b4 ais4 b4 fis4 
  fis2. e4 
  cis4 e4 e4 fis4 
  e4 fis4 gis4 fis4 
  e4 e8 fis8 gis4 a4   %15
  e2~ e8 d8 d8 cis8 
  cis2. 
}

tenor = \relative c' {
  \partial 4 a4 
    a8 b8 cis4 d8 cis8 b4 
  b4 a16 gis16 a8 gis4 e'4 
  e4 e8 dis8 cis8 b16 a16 b8 cis8 
  b4 a16 gis16 a8 gis4 gis4 
  a8 g8 fis8 e8 fis4 fis4   %5
  gis8. a16 b8 a8 gis4 b4 
  a4 b4 cis4 b8 dis8 
  e4 b4 gis4 cis4 
  a8 b8 cis8 d8 e4 e4 
  d4 cis8 d8 e4 e4   %10
  fis4 e4 fis8 e16 d16 cis4 
  d2. gis,4 
  fis4 gis4 a4 a8 b8 
  cis4 b8 a8 e'4 a,8 b8 
  cis8 d8 e4 b4 a4~   %15
  a4 a2 gis4 
  e2. 
}

baixo = \relative c {
  \partial 4 fis4 
    cis4 a4 d4 dis4 
  e4 b4 e4 e4 
  a,4 e'4 fis4 gis8 a8 
  b4 b,4 e4 d4 
  cis4 c4 b8 cis8 dis4   %5
  e4 b4 e4 gis4 
  a4 gis4 cis,8 dis8 e4 
  a,4 b4 e4 cis4 
  fis8 gis8 a4 e8 d8 cis4 
  d4 a4 e'4 cis4   %10
  b4 cis4 d8 e8 fis4 
  b,2. cis4 
  fis4 e4 a4 fis4 
  cis4 d4 e4 fis8 gis8 
  a4 gis8 fis8 e8 d8 cis8 b8   %15
  a8 b8 cis8 d8 e4 e4 
  a,2. 
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