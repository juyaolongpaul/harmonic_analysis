\version "2.10.33"

\header {
  title = "351 - Wenn mein Stündlein vorhanden ist"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key a \major
}


soprano = \relative c'' {
  \partial 4 a4 
    e4 fis4 gis4 a4 
  b4 cis4 a4 cis8 d8 
  e4 e4 cis8 d8 e4 
  d2 cis4 cis4 
  d4 cis4 b4 a8 b8   %5
  cis4 cis4 a4 cis8 d8 
  e4 e4 cis8 d8 e4 
  d2 cis4 cis4 
  d4 cis4 b4 a8 b8 
  cis4 cis4 a4 b4   %10
  cis4 cis4 fis,4 gis4 
  a8 gis8 fis4 e4 e4 
  a4 b4 cis4. b16 cis16 
  d4 cis4 b2 
  a2.   %15
  
}

alto = \relative c' {
  \partial 4 cis8  d8 
    e4 b4 b4 a4 
  e'4 e4 d4 e4 
  e4 e4 e4 fis4 
  fis8 eis8 fis8 gis8 a4 a4 
  b4 a4 gis4 fis4   %5
  fis4. eis8 cis4 fis4 
  e8 fis8 gis4 a4. a8 
  a8 gis16 fis16 gis4 a4 a4 
  a8 gis8 a4 e4. fis16 gis16 
  a4 g4 fis4 e4   %10
  e8 gis8 fis8 e8 dis4 e4 
  fis8 e4 dis8 e4 b4 
  e4 e4 e8 d16 e16 fis4~ 
  fis8 gis8 a4. gis16 fis16 gis4 
  e2.   %15
  
}

tenor = \relative c {
  \partial 4 e4 
    a4 gis8 fis8 cis'4 cis4 
  b4 a8 g8 fis4 a4 
  gis8 a8 b4 a8 b8 cis4 
  b8 cis8 d8 e8 fis4 e4 
  fis8 e4 d4 cis8 cis8 b8   %5
  a4 gis4 fis4 a4 
  b8 a8 b8 e8 e4. e8 
  fis8 d8 b8 e8 e4 e4 
  fis8 b,8 cis8 d8 e4. d8 
  e4 e4 d4 b4   %10
  a8 gis8 a8 cis8 b4 b4 
  b4 cis8 fis,8 gis4 gis8 a16 b16 
  e,4 b'4 a4. gis16 a16 
  b8 b8 cis8 d16 e16 fis8 d8 b8 e8 
  cis2.   %15
  
}

baixo = \relative c {
  \partial 4 a8  b8 
    cis4 dis4 eis4 fis4 
  gis4 a4 d,4 a'8 b8 
  cis8 b8 a8 gis8 a4 ais4 
  b4 b,4 fis'4 a4~ 
  a8 gis4 fis4 eis8 fis8 d8   %5
  a8 b8 cis4 fis,4 fis'4 
  gis8 fis8 e4 a4 cis,4 
  b4 e4 a,4 a'4~ 
  a2~ a8 gis8 cis8 b8 
  a8 a,4 b16 cis16 d4 gis,4   %10
  a8 e'8 a4. gis8 fis8 e8 
  dis8 e8 a8 b8 e,4 e8 d8 
  cis8 b8 a8 gis8 a8 b16 cis16 d4~ 
  d8 cis16 b16 fis'8 e8 d8 b8 e4 
  a,2.   %15
  
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