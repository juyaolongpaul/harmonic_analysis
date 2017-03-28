\version "2.10.33"

\header {
  title = "237 - Was betrübst du dich, mein Herze"
  composer = "J. S. Bach"
}


global = {
  \time 4/4
  \key d \minor
}


soprano = \relative c'' {
  d4 c4 bes4 a4 
  bes4 a8 g8 fis4 d4 
  d'4 d4 ees4 c4 
  d4 c8 bes8 a2 
  d4 c4 bes4 a4   %5
  bes4 a8 g8 fis4 d4 
  d'4 d4 ees4 c4 
  d4 c4 bes2 
  d4 ees4 f4 d4 
  ees4 d4 d8 c8 c4   %10
  c4 d4 ees4 d4 
  c4. bes8 a4 f4 
  d'4 a4 bes4 a4 
  bes4 c4 d2 
  d4 d4 c4 bes4   %15
  a4. g8 g2 
  
}

alto = \relative c'' {
  g8 a16 bes16 a4~ a8 g~ g fis
  d8 g16 f16 ees4 d a 
  bes8 c8 d4 g,4 g'4 
  a4. g8 fis2 
  fis4 g8 a4 g4 fis8   %5
  g8 f8 e4 d4 a4 
  bes8 d8 f4 g4. f8 
  f4 f8. ees16 d2 
  f4 g4 f4 g8 f8 
  ees8 c8 f4 f4 f4   %10
  f4 f4 g8 fis8 g4 
  g8 fis8 g4 f4 c4 
  a'8 g8 a4. g4 fis8 
  g4 g4 fis2 
  g4 g4. fis8 g4~   %15
  g8 fis16 e16 fis4 d2 
  
}

tenor = \relative c' {
  bes8 c16 d16 ees8 d8 d4~ d8. c16 
  bes8 d8 c8 bes8 a4 fis4 
  g4 g'8 f8 ees16 d16 c4 bes8 
  a8 d8 d4 d2 
  a4 g8 d'8 d4 d4   %5
  d4 e8 a,8 a4 fis4 
  f8 bes8 d8 c8 bes4 c4~ 
  c8 bes4 a8 f2 
  d'8 c8 bes4. a8 bes4~ 
  bes8 a8 bes4. a8 a4   %10
  a4 bes4. a8 bes4 
  c4 c4 c4 a4 
  a4 d4 d4 d4 
  d4 ees4 a,2 
  bes4 d4 ees8 d8 e4   %15
  a,8 d8 d8. c16 bes2 
  
}

baixo = \relative c' {
  g4. fis8 g4 d4 
  g,4 c4 d4 d,4 
  g8 a8 bes8 g8 c8 d8 ees4 
  fis,4 g4 d'2 
  d4 e8 fis8 g4 d4   %5
  g4 cis,4 d4 d,4 
  bes'4 bes'8 a8 g4 a4 
  bes4 f4 bes,2 
  bes'8 a8 g4 d4 g4 
  c,4 d8 ees8 f4 f,4   %10
  f'8 ees8 d4 c4 g'8 f8 
  ees4 e4 f4 f,4 
  fis'8 e8 fis8 d8 g4 d4 
  g8 f8 ees4 d2 
  g4 bes,4 c4 cis4   %15
  d4 d,4 g2 
  
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