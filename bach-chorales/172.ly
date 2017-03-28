\version "2.10.33"

\header {
  title = "172 - Sei gegrüßet, Jesu gütig"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key g \minor
}


soprano = \relative c'' {
  g4 a4 bes4 bes4 
  c4 bes4 a4 a4 
  bes4 c4 d4 d4 
  ees4 d8 c8 c2 
  bes2 d4 c4   %5
  d4 d4 c4 bes4 
  c4 c4 c4 d4 
  ees4 ees4 d4 c8 d8 
  d2 c2 
  g4 a4 bes4 bes4   %10
  c4 c4 d4 d4 
  d4 d4 ees4 d4 
  c4 bes8 a8 a2 
  g1 
  
}

alto = \relative c' {
  d4 d4 d4 d4 
  g8 fis8 g4 g4 fis4 
  g4 f4 f4 f4 
  ees8 f8 g4 f2 
  f2 f4 f4   %5
  f4 bes4 bes8 a8 bes4 
  bes4 a4 a4 g4 
  g4 g4 g8 f8 ees4 
  aes4 g8 f8 ees2 
  ees4 d4 d8 ees8 f4   %10
  g4 f4 f4 f4 
  d4 g4 g8 c4 b8 
  c8 fis,8 g4 g4 fis4 
  d1 
  
}

tenor = \relative c' {
  bes4 a4. g16 fis16 g4 
  g8 a8 bes8 g8 d'4 d4 
  d4 c4. bes16 a16 bes4 
  bes4 bes4 bes4 a4 
  d2 bes4 c4~   %5
  c4 bes4 g'8 ees8 f4 
  f4 f4 f4 f4 
  ees8 d8 c4 b4 c4 
  c4 b4 g2 
  c4 c4 bes4 bes4~   %10
  bes4 a4 bes4 bes4 
  a8 c8 c8 bes4 aes16 g16 f4 
  g8 a8 bes8 ees8 d4. c8 
  b1 
  
}

baixo = \relative c' {
  g4 fis4 g4. f8 
  ees4 d8 c8 d4 d4 
  g4 a4 bes4. a8 
  g4 ees4 f4 f,4 
  bes2 bes'4 a4   %5
  bes8 a8 g8 f8 ees8 c8 d8 bes8 
  f'4 f,4 f'4 b,4 
  c8 d8 ees8 f8 g4 aes8 g8 
  f4 g4 c,2 
  c'4 fis,4 g4 d4   %10
  ees4 f4 bes,2 
  fis4 g4 c4 d4 
  ees4 d8 c8 d4 d,4 
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