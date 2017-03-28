\version "2.10.33"

\header {
  title = "52 - Wenn mein Stündlein vorhanden ist"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 a4 
    e4 fis4 gis4 a4 
  b4 cis8 b8 a4 cis8 d8 
  e4 e8 d8 cis8 d8 e4 
  d2 cis4 cis4 
  d4 cis4 b4 a8 b8   %5
  cis4 cis4 a4 cis8 d8 
  e4 e4 cis8 d8 e4 
  d2 cis4 cis4 
  d4 cis4 b4 a8 b8 
  cis4 cis4 a4 b4   %10
  cis4 cis4 fis,4 gis4 
  a8 gis8 fis4 e4 e4 
  a4 b4 cis4 b8 cis8 
  d4 cis4 b2 
  a2.   %15
  
}

alto = \relative c' {
  \partial 4 e4 
    e4. dis8 e4 e8 fis8 
  gis8 fis8 gis4 fis4 a4 
  b8 a8 gis8 fis8 e4 e4 
  fis4 e4 e4 e4 
  b'4 e,4 e8 d8 fis e   %5
  e4 g16 fis g8 fis4 a
  gis8 a8 b4 a8 b8 cis4~ 
  cis8 b16 ais16 b4 ais4 gis4 
  fis4. e8 d4 cis4 
  cis4 cis4 cis4 e4   %10
  e4 fis4 fis8 dis8 e4~ 
  e4 dis4 b4 cis4 
  cis8 d4 e8 e8 fis8 gis8 a8 
  a2. gis4 
  e2.   %15
  
}

tenor = \relative c' {
  \partial 4 cis4 
    b4 a4 b4 cis4 
  d4 cis4 cis4 fis4 
  e4 b4 b4 a4 
  a4 gis4 a4 a4 
  a8 gis8 a4 gis8 gis'8 cis, b   %5
  a a' e4 d e
  e4 e4 e8 d8 cis8 fis8 
  fis4. eis8 fis4 cis8 b8 
  a8 b8 cis4 fis,8 gis8 a4 
  gis4 gis4 fis4 gis4   %10
  a4 a4 b4. b8 
  a16 b16 cis4 b16 a16 gis4 gis4 
  fis8 fis'4 e16 d16 cis8 b16 cis16 d8 e8 
  a,8 fis'4 fis8 b,8 cis16 d16 e8 d8 
  cis2.   %15
  
}

baixo = \relative c' {
  \partial 4 a4 
    gis4 fis4 e8 d8 cis4 
  b4 f4 fis4 fis'4 
  gis8 fis8 e4 a4 cis,4 
  b4 e4 a,4 a4 
  b4 cis8 d8 e4 fis8 gis8   %5
  a4 a,4 d4 a'8 b8 
  cis4 gis4 a4 ais4 
  b4 b,4 fis'4 eis4 
  fis8 gis8 a4 d,8 e8 fis4~ 
  fis8 eis16 dis16 eis8 cis8 fis4 e4   %10
  a8 gis8 fis8 e8 dis8 b8 e8 dis8 
  cis8 a8 b4 e4 cis4 
  fis4 gis4 a4. gis8 
  fis4 e8 d8 e4 e,4 
  a2.   %15
  
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