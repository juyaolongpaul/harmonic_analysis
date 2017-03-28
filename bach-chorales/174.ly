\version "2.10.33"

\header {
  title = "174 - Jesus Christus, unser Heiland, der den Tod"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  g4 f4 f4 g4 
  a4 g4 f4 e4 
  d2 r4 f4 
  g4 a4 g4 fis4 
  g2 r4 c4   %5
  g8 a8 bes4 a4 g4 
  a2 r4 g4 
  f4 g4 a4 g4 
  f4 e4 d4 f4 
  g4 a4 g4 fis4   %10
  g1 
  
}

alto = \relative c' {
  d4 d4 d4. c8 
  c8 f4 e8 d4. cis8 
  a2 r4 d4 
  d4 ees8 d8 d4 d4 
  d2 r4 g8 f8   %5
  ees4 d8 g4 fis8 g8 g,8 
  d'2 r4 d8 cis8 
  d8 f4 e8 f4 e4~ 
  e8 d4 cis8 a4 d4 
  d4 ees4 d4 d4   %10
  d1 
  
}

tenor = \relative c' {
  bes4 bes4 a8 bes8 g4 
  f8 a8 bes4 b4 e,8 a16 g16 
  f2 r4 a4 
  bes8 g8 c4. bes8 a4 
  bes2 r4 c4   %5
  c4 d4 d4. c8 
  fis,2 r4 bes4 
  a4 c4 c4 bes4 
  a4~ a8. g16 f4 a4 
  bes4 c4. bes8 a8 c8   %10
  b1 
  
}

baixo = \relative c {
  g8 a8 bes8 c8 d4 e4 
  f4 g4 gis4 a8 a,8 
  d2 r4 d8 c8 
  bes4 fis4 g4 d'4 
  g,2 r4 ees'8 d8   %5
  c4 g4 d'4 ees4 
  d2 r4 g,4 
  d'4 c4 f8 e8 d8 cis8 
  d4 a4 d,4 d'8 c8 
  bes8 a8 g8 fis8 g4 d4   %10
  g1 
  
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