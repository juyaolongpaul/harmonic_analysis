\version "2.10.33"

\header {
  title = "210 - Christe, du Beistand deiner Kreuzgemeine"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key c \major
}


soprano = \relative c' {
  \partial 4 d4 
    d4 e4 f2 
  e2 a4 b4 
  c4 d8 c8 b2 
  a2 r4 c4 
  b4 a4 g4 g4   %5
  a4 g4 f4 f4 
  e2 f4 e4 
  e4 f4 g2 
  e2 a4 b4 
  c4 d8 c8 b2   %10
  a2 r4 fis4 
  g4 g4 g4 fis4 
  g2 r4 e4 
  f4 g4 e2 
  d2.   %15
  
}

alto = \relative c' {
  \partial 4 a4 
    a4 e'4 e4 d4 
  cis2 e4 e4 
  a4 a4 a4 gis4 
  e2 r4 a4~ 
  a8 g8 f8 e8 d4 g4~   %5
  g8 f4 e4 d8 c8 d8 
  c2 c4 cis8 d8 
  e4. d4 c8 d8 b8 
  c2 a'4 a4~ 
  a8 gis8 a4 a4 gis4   %10
  e2 r4 c4 
  d4 d4 ees4 d4 
  d2 r4 cis4 
  d4 d2 cis4 
  a2.   %15
  
}

tenor = \relative c {
  \partial 4 f8  g8 
    a4 a4 a2 
  a2 a4 e'4 
  e4 d8 e8 f4 e8 d8 
  c2 r4 e4 
  f8 e16 d16 c4 b4 c4   %5
  c4 c4 a4 a4 
  g8 a8 bes4 a4 a4 
  a4 a4 g2 
  g2 e'4 f4 
  e4 d8 e8 f4 e8 d8   %10
  c2 r4 a4 
  g4. f8 ees8 bes'8 a8 c8 
  bes2 r4 a4 
  a4 g8 bes8 a8 e8 a8 g8 
  fis2.   %15
  
}

baixo = \relative c {
  \partial 4 d8  e8 
    f8 e8 d8 cis8 d8 e8 f8 d8 
  a2 c8 b8 a8 gis8 
  a8 c8 f8 e8 d8 b8 e4 
  a,2 r4 a4 
  d8 e8 f2 e4   %5
  f4 c4 d4 a8 bes8 
  c2 f,4 a4 
  cis4 d8 c8 b8 a8 b8 g8 
  c2 c4 d4 
  e4 f8 e8 d4 e4   %10
  a,2 r4 a4 
  bes4 b4 c4 d4 
  g,2 r4 a4 
  d8 c8 bes8 g8 a2 
  d2.   %15
  
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